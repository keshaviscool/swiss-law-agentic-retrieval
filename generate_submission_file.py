import modal
import os

app = modal.App("swiss-law-final")

# ----------------------------
# VOLUME
# ----------------------------
volume = modal.Volume.from_name("law-data")

# ----------------------------
# IMAGE
# ----------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "sentence-transformers",
        "faiss-cpu",
        "pandas",
        "numpy",
        "peft",
        "accelerate",
        "langchain-community",
        "tqdm"
    )
)

# ----------------------------
# FUNCTION
# ----------------------------
@app.function(
    image=image,
    gpu="B200",
    volumes={"/data": volume},
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("hfsecret")],
)
def run():
    import torch
    import pandas as pd
    from tqdm.auto import tqdm
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer as HFTokenizer,
    )
    from peft import PeftModel
    from sentence_transformers import CrossEncoder
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from huggingface_hub import login

    # ----------------------------
    # HF LOGIN
    # ----------------------------
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)

    # ----------------------------
    # PATHS
    # ----------------------------
    FAISS_PATH = "/data/faiss_index_new"
    TEST_PATH = "/data/test.csv"
    OUTPUT_PATH = "/data/submission.csv"

    # ----------------------------
    # EMBEDDINGS + FAISS
    # ----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # ----------------------------
    # RERANKER
    # ----------------------------
    reranker = CrossEncoder(
        "keshavsharma/reranker-bge-swiss-law",
        device="cuda",
        max_length=512,
    )

    # ----------------------------
    # LLM (YOUR EXACT LOGIC)
    # ----------------------------
    BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    ADAPTER_PATH = "keshavsharma/llama-3.1-7b-swiss-law-adapter"

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=hf_token,
        trust_remote_code=True,
    )

    # ✅ FIX (your crash)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, ADAPTER_PATH, token=hf_token)
    model.eval()

    # ----------------------------
    # QUERY GENERATION (UNCHANGED)
    # ----------------------------
    def generate_queries(case_query: str, temperature: float = 0.4):
        messages = [
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
            {"role": "user", "content": case_query},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=124,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        queries = [q.strip() for q in generated.strip().split("\n") if q.strip()]
        return queries[:3]

    # ----------------------------
    # TRANSLATION (UNCHANGED)
    # ----------------------------
    TRANSLATION_MODEL_CANDIDATES = [
        "facebook/nllb-200-distilled-1.3B",
        "facebook/nllb-200-distilled-600M",
    ]

    translator_model = None
    translator_tokenizer = None

    for model_id in TRANSLATION_MODEL_CANDIDATES:
        try:
            translator_tokenizer = HFTokenizer.from_pretrained(model_id)
            translator_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            break
        except Exception as e:
            print(f"Failed {model_id}: {e}")

    def translate_en_to_de(texts, batch_size=8):
        translator_tokenizer.src_lang = "eng_Latn"
        forced_bos_token_id = translator_tokenizer.convert_tokens_to_ids("deu_Latn")

        outputs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            inputs = translator_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(translator_model.device)

            with torch.no_grad():
                tokens = translator_model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=256,
                    num_beams=4,
                )

            outputs.extend(
                translator_tokenizer.batch_decode(tokens, skip_special_tokens=True)
            )

        return outputs

    # ----------------------------
    # PIPELINE CONFIG
    # ----------------------------
    CFG = {
        "temperature": 0.7,
        "score_threshold": 0.95,
        "K_RETRIEVE": 1000,
        "max_citations": 50,
    }

    # ----------------------------
    # PREDICTION FUNCTION (UNCHANGED)
    # ----------------------------
    def predict(query_de):
        queries = generate_queries(query_de, CFG["temperature"])
        if not queries:
            queries = [query_de]

        all_docs = {}
        for q in queries:
            for doc in vectorstore.similarity_search(q, k=CFG["K_RETRIEVE"]):
                cit = doc.metadata.get("citation", "")
                if cit not in all_docs:
                    all_docs[cit] = doc

        candidates = list(all_docs.values())
        if not candidates:
            return []

        pairs = [(query_de, doc.page_content) for doc in candidates]
        scores = reranker.predict(pairs)

        cit_scores = sorted(
            zip([d.metadata.get("citation", "") for d in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )

        sorted_scores = [s for _, s in cit_scores]

        if len(sorted_scores) > 5:
            drops = [
                sorted_scores[i] - sorted_scores[i + 1]
                for i in range(min(CFG["max_citations"], len(sorted_scores) - 1))
            ]
            cutoff = drops.index(max(drops[3:])) + 1 if drops[3:] else CFG["max_citations"]
        else:
            cutoff = CFG["max_citations"]

        result = []
        seen = set()

        for i, (cit, score) in enumerate(cit_scores):
            if i >= cutoff or score < CFG["score_threshold"]:
                break
            if cit not in seen:
                seen.add(cit)
                result.append(cit)

        return result

    # ----------------------------
    # RUN
    # ----------------------------
    df = pd.read_csv(TEST_PATH)

    translated = translate_en_to_de(df["query"].astype(str).tolist())

    outputs = []
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
        preds = predict(translated[i])

        outputs.append({
            "query_id": row.query_id,
            "predicted_citations": ";".join(preds),
        })

    pd.DataFrame(outputs).to_csv(OUTPUT_PATH, index=False)

    print("DONE")


@app.local_entrypoint()
def main():
    run.remote()