import modal
import json
import os

# ── Modal setup ───────────────────────────────────────────────────────────────
app = modal.App("reranker-dataset-builder")

# Persistent volume — stores the checkpoint JSON so you can resume any time
volume = modal.Volume.from_name("reranker-samples", create_if_missing=True)

VOLUME_PATH = "/data"
SAVE_PATH   = f"{VOLUME_PATH}/reranker_samples.json"
ERROR_PATH  = f"{VOLUME_PATH}/reranker_errors.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "transformers==4.44.2",
        "peft==0.12.0",
        "accelerate==0.33.0",
        "sentence-transformers==3.0.1",
        "langchain-community==0.2.10",
        "langchain==0.2.10",
        "langchain-huggingface==0.0.3",
        "faiss-cpu==1.8.0",
        "huggingface-hub==0.24.0",
        "pandas",
        "tqdm",
        "sentencepiece",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)


@app.function(
    image=image,
    gpu="H100",          # plenty for inference-only Llama 8B + FAISS retrieval
    timeout=60 * 180,    # 3 hours
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("hfsecret")],
)
def build_dataset(
    train_rows:  list[dict],   # serialised train_csv rows
    laws_rows:   list[dict],   # serialised laws_de rows
    faiss_files: dict[str, bytes],  # filename -> bytes of the FAISS index files
    k_retrieve:  int = 200,
    neg_ratio:   int = 3,
):
    from dataclasses import dataclass
    import gc
    import numpy as np
    import torch
    import pandas as pd
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    # ── Write FAISS index files to /tmp so langchain can load them ────────────
    FAISS_DIR = "/tmp/faiss_index"
    os.makedirs(FAISS_DIR, exist_ok=True)
    for fname, fbytes in faiss_files.items():
        with open(os.path.join(FAISS_DIR, fname), "wb") as f:
            f.write(fbytes)

    # ── Reconstruct dataframes ────────────────────────────────────────────────
    train_csv = pd.DataFrame(train_rows)
    laws_de   = pd.DataFrame(laws_rows)

    # ── Resume logic ──────────────────────────────────────────────────────────
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            saved = json.load(f)
        resume_from = saved["last_completed_idx"] + 1
        raw_samples = saved["samples"]
        print(f"Resuming from row {resume_from}, already have {len(raw_samples)} samples")
    else:
        raw_samples  = []
        resume_from = 0
        print("Starting fresh")

    error_log = []

    def save_checkpoint(idx):
        with open(SAVE_PATH, "w") as f:
            json.dump({"last_completed_idx": idx, "samples": raw_samples}, f)
        volume.commit()   # flush to persistent storage immediately

    def flush_gpu():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @dataclass
    class RetrievedDoc:
        page_content: str
        metadata: dict

    def build_fallback_retriever():
        import faiss
        from sentence_transformers import SentenceTransformer

        faiss_path = os.path.join(FAISS_DIR, "index.faiss")
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"Missing FAISS index file: {faiss_path}")

        # The index was built from non-empty laws_de rows in CSV order.
        laws_non_empty = laws_de[
            laws_de["text"].fillna("").astype(str).str.strip().ne("")
        ].reset_index(drop=True)

        index_texts = laws_non_empty["text"].astype(str).tolist()
        index_cits = laws_non_empty["citation"].fillna("").astype(str).tolist()

        index = faiss.read_index(faiss_path)
        ntotal = index.ntotal
        if ntotal > len(index_texts):
            raise RuntimeError(
                "FAISS index has more vectors than non-empty laws_de rows "
                f"({ntotal} > {len(index_texts)})."
            )
        if ntotal < len(index_texts):
            print(
                "Warning: laws_de has extra non-empty rows compared with FAISS index "
                f"({len(index_texts)} > {ntotal}); truncating mapping to index size."
            )
            index_texts = index_texts[:ntotal]
            index_cits = index_cits[:ntotal]

        embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")

        def retrieve(query_text: str, k: int) -> list[RetrievedDoc]:
            q_vec = embedder.encode(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            _, idxs = index.search(q_vec, k)

            docs = []
            for doc_idx in idxs[0]:
                if doc_idx < 0 or doc_idx >= len(index_texts):
                    continue
                docs.append(
                    RetrievedDoc(
                        page_content=index_texts[doc_idx],
                        metadata={"citation": index_cits[doc_idx]},
                    )
                )
            return docs

        return retrieve

    # ── Load embeddings + FAISS ───────────────────────────────────────────────
    print("Loading embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},   # keep FAISS on CPU, Llama on GPU
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        def retrieve_docs(search_query: str, k: int):
            return vectorstore.similarity_search(search_query, k=k)

        print("FAISS loaded via LangChain docstore.")
    except Exception as e:
        print(f"LangChain FAISS load failed ({type(e).__name__}: {e}).")
        print("Falling back to raw index.faiss + laws_de row mapping...")
        retrieve_docs = build_fallback_retriever()
        print("FAISS loaded via raw index fallback.")

    # ── Load Llama + adapter ──────────────────────────────────────────────────
    BASE_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
    ADAPTER_PATH = "keshavsharma/llama-3.1-7b-swiss-law-adapter"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, token=os.environ["HF_TOKEN"], trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
        trust_remote_code=True,
    )
    print("Loading adapter...")
    llama = PeftModel.from_pretrained(base, ADAPTER_PATH)
    llama.eval()
    print("Model ready.")

    # ── Query generation ──────────────────────────────────────────────────────
    def generate_queries(case_query: str, temperature: float = 0.7) -> list[str]:
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
        text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(llama.device)
        with torch.no_grad():
            outputs = llama.generate(
                **inputs,
                max_new_tokens=128,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return [q.strip() for q in generated.strip().split("\n") if q.strip()][:3]

    # ── Main loop ─────────────────────────────────────────────────────────────
    for idx, row in tqdm(train_csv.iterrows(), total=len(train_csv)):
        if idx < resume_from:
            continue

        query     = row["query"]
        gold_cits = [c.strip() for c in str(row["gold_citations"]).split(";") if c.strip()]

        try:
            flush_gpu()
            llama_queries = generate_queries(query)
            flush_gpu()

            candidate_docs = []
            seen_cits      = set()

            for lq in llama_queries:
                try:
                    docs = retrieve_docs(lq, k_retrieve)
                except Exception as e:
                    print(f"  [Row {idx}] retrieval error: {e}")
                    flush_gpu()
                    continue
                for d in docs:
                    cit = d.metadata.get("citation", "").strip()
                    if cit and cit not in seen_cits:
                        candidate_docs.append(d)
                        seen_cits.add(cit)
                flush_gpu()

            if not candidate_docs:
                error_log.append({"idx": idx, "reason": "no candidates"})
                save_checkpoint(idx)
                continue

            # Positives
            row_samples = []
            for cit in gold_cits:
                match = laws_de[laws_de["citation"] == cit]["text"].values
                if len(match):
                    row_samples.append({"texts": [query, match[0][:512]], "label": 1.0})

            if not row_samples:
                error_log.append({"idx": idx, "reason": "no positives in laws_de"})
                save_checkpoint(idx)
                continue

            # Hard negatives
            non_gold = [
                d for d in candidate_docs
                if d.metadata.get("citation", "").strip() not in gold_cits
            ]
            for doc in non_gold[: len(row_samples) * neg_ratio]:
                row_samples.append({"texts": [query, doc.page_content[:512]], "label": 0.0})

            raw_samples.extend(row_samples)

            if idx % 10 == 0:
                save_checkpoint(idx)
                pos = sum(1 for s in raw_samples if s["label"] == 1.0)
                neg = sum(1 for s in raw_samples if s["label"] == 0.0)
                print(f"  [Row {idx}] pos={pos} neg={neg} total={len(raw_samples)}")

        except torch.cuda.OutOfMemoryError:
            flush_gpu()
            print(f"  [Row {idx}] OOM — skipping")
            error_log.append({"idx": idx, "reason": "OOM"})
            save_checkpoint(idx)

        except Exception as e:
            print(f"  [Row {idx}] {type(e).__name__}: {e} — skipping")
            error_log.append({"idx": idx, "reason": f"{type(e).__name__}: {e}"})
            save_checkpoint(idx)

    # Final save
    save_checkpoint(len(train_csv) - 1)
    with open(ERROR_PATH, "w") as f:
        json.dump(error_log, f, indent=2)
    volume.commit()

    pos = sum(1 for s in raw_samples if s["label"] == 1.0)
    neg = sum(1 for s in raw_samples if s["label"] == 0.0)
    print(f"\nDone. total={len(raw_samples)} | pos={pos} | neg={neg} | skipped={len(error_log)}")


@app.local_entrypoint()
def main():
    import pandas as pd

    # ── Load local files ──────────────────────────────────────────────────────
    FAISS_DIR = "../faiss_index_new"   # adjust to your local path
    TRAIN_CSV = "../data/train.csv"
    LAWS_CSV  = "../data/laws_de.csv"

    print("Reading CSVs...")
    train_csv = pd.read_csv(TRAIN_CSV)
    laws_de   = pd.read_csv(LAWS_CSV)

    print("Reading FAISS index files...")
    faiss_files = {}
    for fname in os.listdir(FAISS_DIR):
        fpath = os.path.join(FAISS_DIR, fname)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                faiss_files[fname] = f.read()
    print(f"  Loaded {len(faiss_files)} FAISS files: {list(faiss_files.keys())}")

    print(f"Sending {len(train_csv)} rows to Modal A10G...")
    build_dataset.remote(
        train_rows=train_csv.to_dict("records"),
        laws_rows=laws_de.to_dict("records"),
        faiss_files=faiss_files,
    )
    print("Job submitted. Monitor with: modal app logs reranker-dataset-builder")
