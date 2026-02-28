import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Number of parallel GPU workers. Each worker gets one B200 and embeds its
# own shard of documents independently. All shards are merged at the end.
NUM_WORKERS = 8

# BGE-M3 produces 1024-dim embeddings. On a B200 (192 GB HBM3e) with
# batch_size=2048 you get ~25-30k chunks/s → 5M docs in ~3-4 min per worker
# → with 8 workers total wall-time ≈ 3-4 min for all 5M docs.
BATCH_SIZE = 2048

app = modal.App("swiss-law-faiss-builder")

volume = modal.Volume.from_name("swiss-law-volume", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "langchain",
        "langchain-community",
        "langchain-text-splitters",
        "sentence-transformers",
        "faiss-cpu",
        #"faiss-gpu",          # GPU-accelerated FAISS for the merge step
        "pandas",
        "numpy",
        "tqdm",
    )
)

# ---------------------------------------------------------------------------
# STEP 1 — Load & split CSVs (single cheap CPU job, runs once)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    cpu=8,
    memory=32768,            # 32 GB RAM – CSVs are large
    timeout=60 * 30,
    volumes={"/data": volume},
)
def prepare_chunks() -> int:
    """
    Reads both CSVs, splits them into chunks, serialises every chunk as a
    plain-text line in /data/chunks.jsonl and returns the total chunk count.
    Subsequent workers read directly from that file – no re-parsing needed.
    """
    import json
    from langchain_community.document_loaders import CSVLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print("Loading CSVs...")
    laws_docs = CSVLoader(file_path="/data/laws_de.csv").load()
    court_docs = CSVLoader(file_path="/data/court_considerations.csv").load()

    print(f"  laws_de:               {len(laws_docs):,} raw docs")
    print(f"  court_considerations:  {len(court_docs):,} raw docs")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_docs = splitter.split_documents(laws_docs + court_docs)
    total = len(all_docs)
    print(f"Total chunks after splitting: {total:,}")

    # Write every chunk to a JSONL file so workers can seek into it cheaply
    print("Writing chunks.jsonl ...")
    with open("/data/chunks.jsonl", "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps({"text": doc.page_content, "meta": doc.metadata}) + "\n")

    volume.commit()   # Flush writes so other functions see the file
    print("prepare_chunks done.")
    return total


# ---------------------------------------------------------------------------
# STEP 2 — Embed one shard (one B200 per worker, spawned N times in parallel)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="B200",              # Nvidia B200 – 192 GB HBM3e, ~2.4 PFLOPs BF16
    timeout=60 * 60,         # 1-hour safety ceiling (should finish in <10 min)
    volumes={"/data": volume},
)
def embed_shard(worker_id: int, total: int, num_workers: int) -> str:
    """
    Each worker reads only its own slice of chunks.jsonl, embeds them on its
    local B200, and saves the raw numpy arrays + metadata to:
        /data/shards/shard_{worker_id}.npz
    Returns the output path so the merge step can find every file.
    """
    import json, time
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from pathlib import Path

    # ---- Determine this worker's slice ----
    shard_size = (total + num_workers - 1) // num_workers   # ceiling division
    start_idx = worker_id * shard_size
    end_idx   = min(start_idx + shard_size, total)
    n_local   = end_idx - start_idx

    print(f"[worker {worker_id}] rows {start_idx:,} – {end_idx:,}  ({n_local:,} chunks)")

    # ---- Read only this worker's lines ----
    texts, metas = [], []
    with open("/data/chunks.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if i >= end_idx:
                break
            obj = json.loads(line)
            texts.append(obj["text"])
            metas.append(obj["meta"])

    # ---- Load model directly on this GPU ----
    # SentenceTransformer is used directly (faster than HuggingFaceEmbeddings
    # wrapper) and supports multi-GPU via device parameter.
    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    model.half()   # BF16 → halves VRAM, ~2× throughput on B200 with no quality loss

    # ---- Embed in large batches ----
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # Required for cosine similarity with FAISS
    )
    elapsed = time.time() - t0
    rate = n_local / elapsed
    print(f"[worker {worker_id}] embedded {n_local:,} chunks in {elapsed:.1f}s  ({rate:.0f} chunks/s)")

    # ---- Save shard ----
    Path("/data/shards").mkdir(parents=True, exist_ok=True)
    out_path = f"/data/shards/shard_{worker_id}.npz"
    np.savez_compressed(
        out_path,
        embeddings=embeddings.astype(np.float32),  # FAISS expects float32
        texts=np.array(texts, dtype=object),
        metas=np.array([json.dumps(m) for m in metas], dtype=object),
    )
    volume.commit()
    print(f"[worker {worker_id}] saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# STEP 3 — Merge all shards into one FAISS index (single GPU job)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    cpu=8,
    memory=65536,            # 64 GB RAM for the merge (5M × 1024 × 4B ≈ 20 GB)
    timeout=60 * 60,
    volumes={"/data": volume},
)
def merge_and_save(shard_paths: list[str]) -> str:
    """
    Loads every shard .npz file, stacks the embeddings, builds a single
    IndexFlatIP (inner-product / cosine) FAISS index and saves it alongside
    the LangChain metadata so the existing retrieval code can load it with
    FAISS.load_local().
    """
    import json
    import numpy as np
    import faiss
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.docstore.document import Document
    from langchain_community.docstore.in_memory import InMemoryDocstore

    print(f"Merging {len(shard_paths)} shards...")

    all_embeddings, all_texts, all_metas = [], [], []

    for path in sorted(shard_paths):         # sorted so ordering is deterministic
        data = np.load(path, allow_pickle=True)
        all_embeddings.append(data["embeddings"])
        all_texts.extend(data["texts"].tolist())
        all_metas.extend([json.loads(m) for m in data["metas"].tolist()])
        print(f"  loaded {path}  ({len(data['embeddings']):,} vectors)")

    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    total, dim = embeddings_matrix.shape
    print(f"Total vectors: {total:,}  dim: {dim}")

    # ---- Build FAISS IndexIVFFlat (ANN) on CPU ----
    # IVF clusters vectors into nlist buckets. At query time only nprobe buckets
    # are searched → ~100x faster than IndexFlatIP with ~95-98% recall.
    # Rule of thumb: nlist = 4 × sqrt(N) → 4 × sqrt(5M) ≈ 9000, capped at 4096
    # for a good speed/recall balance.
    #
    # Query time on Mac CPU after this change:
    #   IndexFlatIP  (old) → 8–15 seconds per query
    #   IndexIVFFlat (new) → 50–150 ms per query   ← ~100x speedup
    nlist  = 4096   # number of Voronoi cells (clusters)
    nprobe = 64     # cells searched per query (higher = more accurate, slower)

    print(f"Building FAISS IndexIVFFlat (nlist={nlist}, nprobe={nprobe}) on CPU...")
    quantizer  = faiss.IndexFlatIP(dim)          # assigns vectors to clusters
    index_final = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # IVF index must be trained on a representative sample before adding vectors
    sample_size = min(500_000, total)
    print(f"Training on {sample_size:,} sample vectors...")
    index_final.train(embeddings_matrix[:sample_size])
    index_final.nprobe = nprobe
    print("Training done.")

    CHUNK = 500_000
    for start in range(0, total, CHUNK):
        index_final.add(embeddings_matrix[start: start + CHUNK])
        print(f"  {min(start + CHUNK, total):,} / {total:,} vectors added")

    print(f"FAISS index size: {index_final.ntotal:,} vectors")

    # ---- Wrap in LangChain FAISS so existing retrieval code works ----
    # We build the docstore and index_to_docstore_id mapping manually.
    print("Building LangChain FAISS wrapper...")
    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},          # merge container has no GPU
        encode_kwargs={"normalize_embeddings": True},
    )
    docs = [Document(page_content=t, metadata=m) for t, m in zip(all_texts, all_metas)]
    index_to_id = {i: str(i) for i in range(total)}
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})

    vectorstore = FAISS(
        embedding_function=embedder,
        index=index_final,
        docstore=docstore,
        index_to_docstore_id=index_to_id,
    )

    print("Saving FAISS index to /data/faiss_index ...")
    vectorstore.save_local("/data/faiss_index")
    volume.commit()

    print("merge_and_save done.")
    return "/data/faiss_index"


# ---------------------------------------------------------------------------
# Orchestrator — ties all three steps together
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    import time

    t_start = time.time()

    # Step 1 – prepare chunks (runs once, CPU only)
    print("=== Step 1: prepare_chunks ===")
    total = prepare_chunks.remote()
    print(f"Total chunks: {total:,}")

    # Step 2 – embed in parallel across NUM_WORKERS B200 GPUs
    print(f"\n=== Step 2: embed_shard  ({NUM_WORKERS} workers × B200) ===")
    # starmap dispatches all workers simultaneously; Modal schedules them in parallel
    shard_paths = list(
        embed_shard.starmap(
            [(wid, total, NUM_WORKERS) for wid in range(NUM_WORKERS)]
        )
    )
    print(f"All shards done: {shard_paths}")

    # Step 3 – merge shards into one FAISS index
    print("\n=== Step 3: merge_and_save ===")
    output_path = merge_and_save.remote(shard_paths)

    elapsed = time.time() - t_start
    print(f"\n✅  Pipeline complete in {elapsed / 60:.1f} min  → {output_path}")


@app.local_entrypoint()
def resume():
    """
    Skip Steps 1 & 2 — shards are already on the volume.
    Jumps straight to Step 3 (merge + build FAISS index).

    Run with:  modal run build_embeddings.py::resume
    """
    import time

    t_start = time.time()

    # The 8 shard paths we know were successfully written
    shard_paths = [f"/data/shards/shard_{i}.npz" for i in range(NUM_WORKERS)]
    print(f"Resuming from existing shards: {shard_paths}\n")

    print("=== Step 3: merge_and_save ===")
    output_path = merge_and_save.remote(shard_paths)

    elapsed = time.time() - t_start
    print(f"\n✅  Resume complete in {elapsed / 60:.1f} min  → {output_path}")
