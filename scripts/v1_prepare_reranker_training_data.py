import modal
import json
import os

app = modal.App("reranker-dataset-builder-v2")

volume = modal.Volume.from_name("reranker-samples-v2", create_if_missing=True)

VOLUME_PATH = "/data"
SAVE_PATH   = f"{VOLUME_PATH}/reranker_samples_v2.json"
ERROR_PATH  = f"{VOLUME_PATH}/reranker_errors_v2.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "transformers==4.44.2",
        "peft==0.12.0",
        "accelerate==0.33.0",
        "sentence-transformers==3.0.1",
        # ── LangChain: pin to same major versions used when building the index ──
        # The FAISS docstore is pickled with pydantic v1 internals; mismatched
        # versions cause the __fields_set__ KeyError at load time.
        "langchain==0.1.20",
        "langchain-community==0.0.38",
        "langchain-core==0.1.52",
        "pydantic==1.10.21",          # force pydantic v1 — index was pickled with it
        "faiss-cpu==1.7.4",           # 1.8.0 changed the index binary format
        "huggingface-hub==0.24.0",
        "pandas",
        "tqdm",
        "sentencepiece",
        "numpy",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 180,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("hfsecret")],
)
def build_dataset(
    train_rows:  list[dict],
    laws_rows:   list[dict],
    faiss_files: dict[str, bytes],
    k_retrieve:  int = 500,   # up from 200 — harder negatives
    neg_ratio:   int = 5,     # up from 3
    cross_neg_per_row: int = 3,  # gold articles from OTHER rows as extra negatives
):
    import gc
    import random
    import numpy as np
    import torch
    import pandas as pd
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    # Use the old import path — matches langchain-community==0.0.38 / pydantic v1
    # which is the same environment the FAISS index was originally pickled in
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    random.seed(42)

    # ── Write FAISS files ─────────────────────────────────────────────────────
    FAISS_DIR = "/tmp/faiss_index"
    os.makedirs(FAISS_DIR, exist_ok=True)
    for fname, fbytes in faiss_files.items():
        with open(os.path.join(FAISS_DIR, fname), "wb") as f:
            f.write(fbytes)

    train_csv = pd.DataFrame(train_rows)
    laws_de   = pd.DataFrame(laws_rows)

    # Pre-build a lookup: citation -> text (avoids repeated df queries)
    cit_to_text = {
        str(row["citation"]).strip(): str(row["text"])
        for _, row in laws_de.iterrows()
        if str(row["text"]).strip()
    }
    print(f"Citation lookup built: {len(cit_to_text)} entries")

    # Pre-collect all gold citations across all rows for cross-row negatives
    all_gold_pool: list[dict] = []  # {"query_idx": int, "cit": str, "text": str}
    for idx, row in train_csv.iterrows():
        for cit in [c.strip() for c in str(row["gold_citations"]).split(";") if c.strip()]:
            if cit in cit_to_text:
                all_gold_pool.append({"query_idx": idx, "cit": cit, "text": cit_to_text[cit]})
    print(f"Cross-row gold pool: {len(all_gold_pool)} entries")

    # ── Resume ────────────────────────────────────────────────────────────────
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            saved = json.load(f)
        resume_from = saved["last_completed_idx"] + 1
        raw_samples = saved["samples"]
        print(f"Resuming from row {resume_from}, have {len(raw_samples)} samples")
    else:
        raw_samples  = []
        resume_from = 0
        print("Starting fresh")

    error_log = []

    def save_checkpoint(idx):
        with open(SAVE_PATH, "w") as f:
            json.dump({"last_completed_idx": idx, "samples": raw_samples}, f)
        volume.commit()

    def flush_gpu():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ── Load FAISS ────────────────────────────────────────────────────────────
    # Strategy: try LangChain first (fast, preserves metadata).
    # If it fails due to pickle/pydantic version mismatch, fall back to
    # raw faiss-cpu + SentenceTransformer — no pickle involved at all.
    print("Loading FAISS...")

    retrieve_docs = None

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS as LCFaiss

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = LCFaiss.load_local(
            FAISS_DIR, embeddings, allow_dangerous_deserialization=True
        )

        def retrieve_docs(query: str, k: int):
            return vectorstore.similarity_search(query, k=k)

        print("FAISS loaded via LangChain.")

    except Exception as lc_err:
        print(f"LangChain FAISS load failed ({type(lc_err).__name__}: {lc_err})")
        print("Falling back to raw faiss + SentenceTransformer...")

        import faiss as raw_faiss
        from sentence_transformers import SentenceTransformer
        from dataclasses import dataclass

        @dataclass
        class _Doc:
            page_content: str
            metadata: dict

        # The index was built from non-empty laws_de rows in order
        laws_nonempty = laws_de[
            laws_de["text"].fillna("").str.strip().ne("")
        ].reset_index(drop=True)

        index_texts = laws_nonempty["text"].astype(str).tolist()
        index_cits  = laws_nonempty["citation"].fillna("").astype(str).tolist()

        faiss_index_path = os.path.join(FAISS_DIR, "index.faiss")
        raw_index = raw_faiss.read_index(faiss_index_path)

        # Truncate mapping to index size if there's a mismatch
        n = raw_index.ntotal
        index_texts = index_texts[:n]
        index_cits  = index_cits[:n]

        _embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")

        def retrieve_docs(query: str, k: int):
            vec = _embedder.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            ).astype(np.float32)
            _, idxs = raw_index.search(vec, k)
            return [
                _Doc(page_content=index_texts[i], metadata={"citation": index_cits[i]})
                for i in idxs[0] if 0 <= i < len(index_texts)
            ]

        print(f"Raw FAISS loaded: {n} vectors, {len(index_texts)} mapped texts.")

    # ── Load Llama + adapter ──────────────────────────────────────────────────
    BASE_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
    ADAPTER_PATH = "keshavsharma/llama-3.1-7b-swiss-law-adapter"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, token=os.environ["HF_TOKEN"], trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading Llama + adapter...")
    base  = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto",
        token=os.environ["HF_TOKEN"], trust_remote_code=True,
    )
    llama = PeftModel.from_pretrained(base, ADAPTER_PATH)
    llama.eval()
    print("Model ready.")

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
                **inputs, max_new_tokens=128, temperature=temperature,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return [q.strip() for q in generated.strip().split("\n") if q.strip()][:3]

    # ── Main loop ─────────────────────────────────────────────────────────────
    for idx, row in tqdm(train_csv.iterrows(), total=len(train_csv)):
        if idx < resume_from:
            continue

        # Always use the full case query — never the short ngram queries
        full_query = row["query"]
        gold_cits  = set(c.strip() for c in str(row["gold_citations"]).split(";") if c.strip())

        try:
            # ── Generate retrieval queries ────────────────────────────────────
            flush_gpu()
            llama_queries = generate_queries(full_query)
            flush_gpu()

            # ── Retrieve candidates ───────────────────────────────────────────
            candidate_docs = {}  # cit -> doc
            for lq in llama_queries:
                try:
                    docs = retrieve_docs(lq, k_retrieve)
                except Exception as e:
                    print(f"  [Row {idx}] retrieval error: {e}")
                    flush_gpu()
                    continue
                for d in docs:
                    cit = d.metadata.get("citation", "").strip()
                    if cit and cit not in candidate_docs:
                        candidate_docs[cit] = d
                flush_gpu()

            if not candidate_docs:
                error_log.append({"idx": idx, "reason": "no candidates"})
                save_checkpoint(idx)
                continue

            row_samples = []

            # ── Positives: (full_query, gold_article_text) ────────────────────
            pos_cits = []
            for cit in gold_cits:
                text = cit_to_text.get(cit, "")
                if text:
                    row_samples.append({
                        "texts": [full_query, text[:512]],
                        "label": 1.0
                    })
                    pos_cits.append(cit)

            if not row_samples:
                error_log.append({"idx": idx, "reason": "no positives in laws_de"})
                save_checkpoint(idx)
                continue

            n_pos = len(row_samples)

            # ── Hard negatives type 1: retrieved but not gold ─────────────────
            # These are the hardest — FAISS thinks they're relevant but they're not
            in_batch_negs = [
                d for cit, d in candidate_docs.items()
                if cit not in gold_cits
            ]
            for doc in in_batch_negs[: n_pos * neg_ratio]:
                row_samples.append({
                    "texts": [full_query, doc.page_content[:512]],
                    "label": 0.0
                })

            # ── Hard negatives type 2: gold articles from OTHER rows ──────────
            # These teach the reranker that relevance is query-specific,
            # not just "this is a law article therefore relevant"
            other_gold = [
                g for g in all_gold_pool
                if g["query_idx"] != idx and g["cit"] not in gold_cits
            ]
            cross_negs = random.sample(other_gold, min(cross_neg_per_row, len(other_gold)))
            for g in cross_negs:
                row_samples.append({
                    "texts": [full_query, g["text"][:512]],
                    "label": 0.0
                })

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

    FAISS_DIR = "../faiss_index_new"
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

    print(f"Sending {len(train_csv)} rows to Modal H100...")
    build_dataset.remote(
        train_rows=train_csv.to_dict("records"),
        laws_rows=laws_de.to_dict("records"),
        faiss_files=faiss_files,
    )
    print("Job submitted.")