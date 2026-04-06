import modal
import json
import os

# ── Modal setup ───────────────────────────────────────────────────────────────
app = modal.App("qwen-swiss-law-finetune")

# Volume to persist the trained adapter
volume = modal.Volume.from_name("swiss-law-adapter-rank64-epoch5", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7.0",
        "transformers==4.44.2",
        "datasets==2.20.0",
        "peft==0.12.0",
        "trl==0.10.1",
        "accelerate==0.33.0",
        "sentencepiece",
        "scipy",
        "rich"
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

# ── Data prep (runs locally before upload) ────────────────────────────────────
def prepare_dataset(json_path: str) -> list[dict]:
    """
    Convert ngram_top3_per_row_dataset.json into chat-format training samples.
    Each row has top3_ngrams — we use ALL 3 as a single assistant turn
    (3 queries separated by newline), keeping only samples where at least
    one ngram hits a gold citation.
    """
    with open(json_path) as f:
        data = json.load(f)

    samples = []
    for row in data:
        query      = row["query"]
        top3       = row["top3_ngrams"]

        # Filter: keep only ngrams that actually hit gold
        good_ngrams = [
            ng["ngram"] for ng in top3
            if ng["gold_hits_unique_bases"] >= 1
        ]

        if not good_ngrams:
            continue

        # Pad to 3 queries (repeat last if fewer than 3 good ones)
        while len(good_ngrams) < 3:
            good_ngrams.append(good_ngrams[-1])
        good_ngrams = good_ngrams[:3]

        target = "\n".join(good_ngrams)

        samples.append({
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Du bist ein Rechtsrecherche-Spezialist für Schweizer Recht. "
                        "Gegeben einen deutschen Rechtsfall, generiere genau 3 kurze deutsche "
                        "Stichwortsuchanfragen, die die relevantesten Gesetzesartikel aus einer "
                        "Vektordatenbank abrufen. "
                        "Gib NUR die 3 Suchanfragen aus, eine pro Zeile, ohne Nummerierung, "
                        "ohne Symbole, ohne weiteren Text."
                    ),
                },
                {
                    "role": "user",
                    "content": query,
                },
                {
                    "role": "assistant",
                    "content": target,
                },
            ]
        })

    return samples


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 90,          # 90 min max
    volumes={"/output": volume},
    secrets=[modal.Secret.from_name("hfsecret")],  # HF_TOKEN
)
def finetune(samples: list[dict]):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    from datasets import Dataset

    MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"
    OUTPUT_DIR = "/output/qwen_swiss_law_adapter"

    print(f"Loaded {len(samples)} training samples")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=os.environ["HF_TOKEN"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Format to string using chat template ──────────────────────────────────
    def format_sample(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(samples)
    dataset = dataset.map(format_sample, remove_columns=["messages"])

    # 90/10 split
    split   = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]
    print(f"Train: {len(train_ds)}  |  Eval: {len(eval_ds)}")

    # B200 (sm_100) often fails with older bitsandbytes 4-bit kernels.
    # Auto-disable 4-bit on Blackwell unless explicitly forced.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this job, but no GPU is visible in the container.")

    gpu_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    is_blackwell = cc_major >= 10

    force_4bit = os.environ.get("FORCE_4BIT", "0").strip().lower() in {"1", "true", "yes"}
    use_4bit = force_4bit or (not is_blackwell)

    print(f"GPU: {gpu_name} | compute capability: sm_{cc_major}{cc_minor}")
    print(f"Using 4-bit bitsandbytes path: {use_4bit} (FORCE_4BIT={force_4bit})")

    # ── Model in 4-bit ────────────────────────────────────────────────────────
    optimizer_name = "adamw_torch"

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                token=os.environ["HF_TOKEN"],
            )
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model = prepare_model_for_kbit_training(model)
            optimizer_name = "paged_adamw_8bit"
        except Exception as e:
            # Typical failure on newer Triton stacks: `No module named triton.ops`.
            print(f"4-bit path unavailable ({type(e).__name__}: {e}); falling back to bf16 full-model LoRA.")
            use_4bit = False

    if not use_4bit:
        # B200-safe fallback: avoid bitsandbytes kernels entirely.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=os.environ["HF_TOKEN"],
        )

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Trainer ───────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,      # effective batch = 16
        per_device_eval_batch_size=1,
        warmup_ratio=0.05,
        learning_rate=2e-4,
        bf16=True,
        optim=optimizer_name,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=1024,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    volume.commit()
    print(f"Adapter saved to {OUTPUT_DIR}")


@app.local_entrypoint()
def main():
    import os

    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../data/ngram_top3_per_row_dataset.json",
    )
    print(f"Loading dataset from: {data_path}")
    samples = prepare_dataset(data_path)
    print(f"Prepared {len(samples)} samples — uploading to Modal B200...")
    finetune.remote(samples)
    print("Training complete.")