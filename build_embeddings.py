import modal

NUM_WORKERS = 8
BATCH_SIZE  = 64

app    = modal.App("swiss-law-faiss-builder")
volume = modal.Volume.from_name("swiss-law-volume", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "langchain",
        "langchain-community",
        "sentence-transformers",
        "faiss-cpu",
        "pandas",
        "numpy",
        "tqdm",
        "fastapi",        # ✅ REQUIRED
        "uvicorn",
    )
)

# ================================
# STEP 1 — PREPARE CHUNKS
# ================================
@app.function(
    image=image,
    cpu=4,
    memory=32768,  # 🔥 increased to 32GB
    timeout=60 * 30,
    volumes={"/data": volume},
)
def prepare_chunks() -> int:
    import json, csv

    rows_written = 0

    def _load(path, cite_col, text_col, tag, out_f):
        nonlocal rows_written
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                citation = row.get(cite_col, "").strip()
                text     = row.get(text_col, "").strip()
                if not text:
                    continue
                out_f.write(json.dumps({
                    "text": text,
                    "meta": {"citation": citation, "source": tag},
                }, ensure_ascii=False) + "\n")
                rows_written += 1

    with open("/data/chunks.jsonl", "w", encoding="utf-8") as out_f:
        _load("/data/laws_de.csv", "citation", "text", "laws_de", out_f)
        # _load("/data/court_considerations.csv", "citation", "text", "court_considerations", out_f)

    volume.commit()
    return rows_written


# ================================
# STEP 2 — EMBEDDING
# ================================
@app.function(
    image=image,
    gpu="B200",  # safer than B200
    timeout=60 * 60,
    volumes={"/data": volume},
)
def embed_shard(worker_id: int, total: int, num_workers: int) -> str:
    import json, numpy as np
    from sentence_transformers import SentenceTransformer
    from pathlib import Path

    shard_size = (total + num_workers - 1) // num_workers
    start_idx  = worker_id * shard_size
    end_idx    = min(start_idx + shard_size, total)

    texts, metas = [], []

    with open("/data/chunks.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_idx: continue
            if i >= end_idx: break
            obj = json.loads(line)
            texts.append(obj["text"])
            metas.append(obj["meta"])

    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    model.half()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    Path("/data/shards").mkdir(parents=True, exist_ok=True)
    out_path = f"/data/shards/shard_{worker_id}.npz"

    np.savez_compressed(
        out_path,
        embeddings=embeddings.astype(np.float32),
        texts=np.array(texts, dtype=object),
        metas=np.array(metas, dtype=object),
    )

    volume.commit()
    return out_path


# ================================
# STEP 3 — MERGE
# ================================
@app.function(
    image=image,
    cpu=8,
    memory=65536,
    timeout=60 * 60,
    volumes={"/data": volume},
)
def merge_and_save(shard_paths: list[str]) -> str:
    import numpy as np, faiss
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.docstore.document import Document
    from langchain_community.docstore.in_memory import InMemoryDocstore

    all_embeddings, all_texts, all_metas = [], [], []

    for path in shard_paths:
        data = np.load(path, allow_pickle=True)
        all_embeddings.append(data["embeddings"])
        all_texts.extend(data["texts"].tolist())
        all_metas.extend(data["metas"].tolist())

    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    total, dim = embeddings_matrix.shape

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, 4096, faiss.METRIC_INNER_PRODUCT)

    index.train(embeddings_matrix[:500000])
    index.nprobe = 64

    index.add(embeddings_matrix)

    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    docs = [Document(page_content=t, metadata=m) for t, m in zip(all_texts, all_metas)]
    docstore = InMemoryDocstore({str(i): d for i, d in enumerate(docs)})
    index_to_id = {i: str(i) for i in range(total)}

    vectorstore = FAISS(
        embedding_function=embedder,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_id,
    )

    vectorstore.save_local("/data/faiss_index_new")
    volume.commit()

    return "/data/faiss_index_new"


# ================================
# STEP 4 — RETRIEVAL API + RERANK
# ================================
vectorstore = None
reranker = None

@app.function(
    image=image,
    cpu=4,
    memory=32768,
    timeout=60,
    volumes={"/data": volume},
)
@modal.web_endpoint(method="POST")
def retrieve(request: dict):
    global vectorstore, reranker

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from sentence_transformers import CrossEncoder

    query = request.get("query")
    k = request.get("k", 20)
    rerank_k = request.get("rerank_k", 5)

    if not query:
        return {"error": "query required"}

    # Load FAISS once
    if vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        vectorstore = FAISS.load_local(
            "/data/faiss_index_new",
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Load reranker once
    if reranker is None:
        reranker = CrossEncoder("BAAI/bge-reranker-base")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    docs = retriever.invoke(query)

    # RERANK
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    final_docs = [doc for doc, _ in ranked[:rerank_k]]

    return {
        "results": [
            {"text": d.page_content, "metadata": d.metadata}
            for d in final_docs
        ]
    }


# ================================
# ENTRYPOINT
# ================================
@app.local_entrypoint()
def main():
    total = prepare_chunks.remote()
    shard_paths = list(
        embed_shard.starmap([(i, total, NUM_WORKERS) for i in range(NUM_WORKERS)])
    )
    merge_and_save.remote(shard_paths)