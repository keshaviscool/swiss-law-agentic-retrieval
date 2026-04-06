import modal
import json
import os

# ── Modal setup ───────────────────────────────────────────────────────────────
app = modal.App("gemma-swiss-law-finetune")

volume = modal.Volume.from_name("gemma-law-adapter-rank32-epoch3", create_if_missing=True)

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

# ── Data prep ────────────────────────────────────────────────────────────────
def prepare_dataset(json_path: str) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)

    samples = []
    for row in data:
        query = row["query"]
        top3 = row["top3_ngrams"]

        good_ngrams = [
            ng["ngram"] for ng in top3
            if ng["gold_hits_unique_bases"] >= 1
        ]

        if not good_ngrams:
            continue

        while len(good_ngrams) < 3:
            good_ngrams.append(good_ngrams[-1])
        good_ngrams = good_ngrams[:3]

        target = "\n".join(good_ngrams).strip()

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
    timeout=60 * 90,
    volumes={"/output": volume},
    secrets=[modal.Secret.from_name("hfsecret")],
)
def finetune(samples: list[dict]):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    from datasets import Dataset

    MODEL_ID = "google/gemma-4-E4B"
    OUTPUT_DIR = "/output/gemma_swiss_law_adapter"

    print(f"Loaded {len(samples)} training samples")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer_kwargs = {
        "token": os.environ["HF_TOKEN"],
        "trust_remote_code": True,
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            use_fast=True,
            **tokenizer_kwargs,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            use_fast=False,
            **tokenizer_kwargs,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # 🔥 FIX: Gemma EOS + chat template
    tokenizer.eos_token = "<end_of_turn>"

    tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}
<start_of_turn>system
{{ message['content'] }}<end_of_turn>
{% elif message['role'] == 'user' %}
<start_of_turn>user
{{ message['content'] }}<end_of_turn>
{% elif message['role'] == 'assistant' %}
<start_of_turn>model
{{ message['content'] }}<end_of_turn>
{% endif %}
{% endfor %}"""

    # ── Format dataset ────────────────────────────────────────────────────────
    def format_sample(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(samples)
    dataset = dataset.map(format_sample, remove_columns=["messages"])

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    # ── Model loading ─────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ["HF_TOKEN"],
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules = ["linear"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # ── Training ──────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        # processing_class=tokenizer,
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
    data_path = "../data/ngram_top3_per_row_dataset.json"
    samples = prepare_dataset(data_path)
    finetune.remote(samples)
