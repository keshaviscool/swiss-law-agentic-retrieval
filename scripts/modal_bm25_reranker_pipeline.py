import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "swiss-law-bm25-reranker"
VOLUME_NAME = "swiss-law-reranker-artifacts"
DATA_VOLUME_NAME = "swiss-law-input-data"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_DIR = PROJECT_ROOT / "data"
REMOTE_DATA_DIR = Path("/data")
REMOTE_ARTIFACT_DIR = Path("/artifacts")

app = modal.App(APP_NAME)
artifacts_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "rank-bm25>=0.2.2",
        "sentence-transformers>=3.0.0",
        "torch>=2.2.0",
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
    )
)

_BM25 = None
_DOC_IDS: list[str] = []
_CITATION_TO_TEXT: dict[str, str] = {}


def _init_bm25_worker() -> None:
    # Worker processes inherit globals via fork; this only validates setup.
    if _BM25 is None or not _DOC_IDS or not _CITATION_TO_TEXT:
        raise RuntimeError("BM25 worker globals were not initialized.")


def _get_bm25_negatives_fast(
    query: str,
    gold_citations: set[str],
    top_k: int,
    num_neg: int,
    candidate_pool_multiplier: int,
) -> list[str]:
    import numpy as np

    tokenized_query = query.lower().split()
    scores = _BM25.get_scores(tokenized_query)

    candidate_count = min(
        len(scores),
        max(num_neg * candidate_pool_multiplier, top_k * 5, 512),
    )

    top_idx_unsorted = np.argpartition(scores, -candidate_count)[-candidate_count:]
    top_idx = top_idx_unsorted[np.argsort(scores[top_idx_unsorted])[::-1]]

    negatives: list[str] = []
    for idx in top_idx:
        citation = _DOC_IDS[int(idx)]
        if citation not in gold_citations:
            negatives.append(citation)
        if len(negatives) >= num_neg:
            return negatives

    # Rare fallback path when high-scoring set is saturated by positives.
    for idx in np.argsort(scores)[::-1]:
        citation = _DOC_IDS[int(idx)]
        if citation not in gold_citations:
            negatives.append(citation)
        if len(negatives) >= num_neg:
            break

    return negatives[:num_neg]


def _process_train_chunk(args: tuple) -> dict:
    (
        chunk_id,
        rows,
        chunk_path,
        bm25_top_k,
        negatives_per_query,
        candidate_pool_multiplier,
    ) = args

    total_positive = 0
    total_negative = 0
    missing_positive = 0

    with Path(chunk_path).open("w", encoding="utf-8") as out:
        for _, query, gold_citations_raw in rows:
            gold_citations = set(str(gold_citations_raw).split(";"))

            for citation in gold_citations:
                if citation in _CITATION_TO_TEXT:
                    out.write(
                        json.dumps(
                            {"query": query, "doc": _CITATION_TO_TEXT[citation], "label": 1},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    total_positive += 1
                else:
                    missing_positive += 1

            negatives = _get_bm25_negatives_fast(
                query=query,
                gold_citations=gold_citations,
                top_k=bm25_top_k,
                num_neg=negatives_per_query,
                candidate_pool_multiplier=candidate_pool_multiplier,
            )

            for citation in negatives:
                out.write(
                    json.dumps(
                        {"query": query, "doc": _CITATION_TO_TEXT[citation], "label": 0},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total_negative += 1

    return {
        "chunk_id": chunk_id,
        "chunk_path": chunk_path,
        "rows": len(rows),
        "total_positive": total_positive,
        "total_negative": total_negative,
        "missing_positive": missing_positive,
    }

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@app.function(
    image=image,
    volumes={str(REMOTE_DATA_DIR): data_volume, str(REMOTE_ARTIFACT_DIR): artifacts_volume},
    cpu=32,
    memory=262_144,
    timeout=60 * 60 * 8,
)
def build_training_data_remote(config: dict) -> dict:
    import json
    import multiprocessing as mp
    from pathlib import Path

    import pandas as pd
    from rank_bm25 import BM25Okapi

    def log(msg: str) -> None:
        print(f"[{_now()}] [BM25] {msg}", flush=True)

    run_id = config["run_id"]
    negatives_per_query = int(config.get("negatives_per_query", 5))
    bm25_top_k = int(config.get("bm25_top_k", 50))
    log_every = int(config.get("log_every", 50))
    num_workers = int(config.get("num_workers", 32))
    chunk_size = int(config.get("chunk_size", 200))
    candidate_pool_multiplier = int(config.get("candidate_pool_multiplier", 32))

    # Avoid CPU oversubscription when using process-level parallelism.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    run_dir = Path("/artifacts") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    laws_de_path = Path("/data/laws_de.csv")
    court_path = Path("/data/court_considerations.csv")
    train_path = Path("/data/train.csv")

    log("Loading CSVs...")
    laws_de = pd.read_csv(laws_de_path)
    court_considerations = pd.read_csv(court_path)
    train = pd.read_csv(train_path)
    log(
        f"Loaded laws_de={len(laws_de):,}, court_considerations={len(court_considerations):,}, train={len(train):,}"
    )

    citation_to_text: dict[str, str] = {}

    log("Building citation -> text map from laws_de...")
    for i, row in laws_de.iterrows():
        citation_to_text[row["citation"]] = f"{row['title']} | {row['text']}"
        if (i + 1) % 200_000 == 0:
            log(f"Processed laws_de rows: {i + 1:,}")

    log("Merging duplicate court_considerations entries into citation -> text map...")
    merged_count = 0
    new_count = 0
    for i, row in court_considerations.iterrows():
        citation = row["citation"]
        if citation in citation_to_text:
            citation_to_text[citation] += f" | {row['text']}"
            merged_count += 1
        else:
            citation_to_text[citation] = row["text"]
            new_count += 1

        if (i + 1) % 200_000 == 0:
            log(f"Processed court_considerations rows: {i + 1:,}")

    log(
        f"Final unique citations={len(citation_to_text):,}, merged_into_existing={merged_count:,}, newly_created={new_count:,}"
    )

    log("Preparing BM25 corpus from all documents...")
    doc_ids: list[str] = []
    corpus: list[list[str]] = []

    for idx, (citation, text) in enumerate(citation_to_text.items(), start=1):
        doc_ids.append(citation)
        corpus.append(str(text).lower().split())
        if idx % 200_000 == 0:
            log(f"Tokenized documents: {idx:,}")

    log("Initializing BM25 index on full corpus...")
    bm25 = BM25Okapi(corpus)
    log("BM25 index ready.")

    global _BM25, _DOC_IDS, _CITATION_TO_TEXT
    _BM25 = bm25
    _DOC_IDS = doc_ids
    _CITATION_TO_TEXT = citation_to_text

    train_data_jsonl = run_dir / "train_data.jsonl"
    total_positive = 0
    total_negative = 0
    missing_positive = 0

    target_tasks = max(1, num_workers * 8)
    adaptive_chunk_size = max(8, len(train) // target_tasks) if len(train) > 0 else chunk_size
    effective_chunk_size = max(8, min(chunk_size, adaptive_chunk_size))

    log(
        f"Generating training pairs in parallel | workers={num_workers}, chunk_size={effective_chunk_size} (requested={chunk_size}), candidate_pool_multiplier={candidate_pool_multiplier}"
    )

    train_rows = [
        (i, row["query"], row["gold_citations"])
        for i, row in train.iterrows()
    ]

    chunks_dir = run_dir / "jsonl_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for chunk_id, start in enumerate(range(0, len(train_rows), effective_chunk_size), start=1):
        rows = train_rows[start : start + effective_chunk_size]
        chunk_path = str(chunks_dir / f"chunk_{chunk_id:06d}.jsonl")
        tasks.append(
            (
                chunk_id,
                rows,
                chunk_path,
                bm25_top_k,
                negatives_per_query,
                candidate_pool_multiplier,
            )
        )

    processed_rows = 0
    chunk_results: list[dict] = []

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=num_workers, initializer=_init_bm25_worker) as pool:
        for chunk_result in pool.imap_unordered(_process_train_chunk, tasks):
            chunk_results.append(chunk_result)
            processed_rows += chunk_result["rows"]
            total_positive += chunk_result["total_positive"]
            total_negative += chunk_result["total_negative"]
            missing_positive += chunk_result["missing_positive"]

            if processed_rows % max(log_every, chunk_size) == 0 or processed_rows >= len(train_rows):
                total = total_positive + total_negative
                log(
                    f"Processed train rows: {processed_rows:,}/{len(train_rows):,} | pairs={total:,} (pos={total_positive:,}, neg={total_negative:,})"
                )

    log("Merging chunked JSONL files...")
    chunk_results.sort(key=lambda x: x["chunk_id"])
    with train_data_jsonl.open("w", encoding="utf-8") as out:
        for res in chunk_results:
            with Path(res["chunk_path"]).open("r", encoding="utf-8") as chunk_file:
                shutil.copyfileobj(chunk_file, out)

    shutil.rmtree(chunks_dir, ignore_errors=True)

    stats = {
        "run_id": run_id,
        "train_rows": int(len(train)),
        "unique_citations": int(len(citation_to_text)),
        "total_positive_pairs": int(total_positive),
        "total_negative_pairs": int(total_negative),
        "missing_positive_pairs": int(missing_positive),
        "training_data_jsonl": str(train_data_jsonl),
    }

    stats_path = run_dir / "build_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    artifacts_volume.commit()
    log(f"Saved training data to {train_data_jsonl}")
    log(f"Saved stats to {stats_path}")

    return stats


@app.function(
    image=image,
    volumes={str(REMOTE_ARTIFACT_DIR): artifacts_volume},
    gpu="B200",
    cpu=16,
    memory=131_072,
    timeout=60 * 60 * 8,
)
def train_reranker_remote(config: dict) -> dict:
    import json
    from pathlib import Path

    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    def log(msg: str) -> None:
        print(f"[{_now()}] [RERANKER] {msg}", flush=True)

    run_id = config["run_id"]
    model_name = config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    epochs = int(config.get("epochs", 3))
    batch_size = int(config.get("batch_size", 128))
    max_train_samples = config.get("max_train_samples")
    max_train_samples = int(max_train_samples) if max_train_samples is not None else None

    run_dir = Path("/artifacts") / run_id
    train_data_jsonl = run_dir / "train_data.jsonl"

    if not train_data_jsonl.exists():
        raise FileNotFoundError(f"Training data not found at {train_data_jsonl}")

    log(f"Loading training pairs from {train_data_jsonl}")
    train_examples: list[InputExample] = []

    with train_data_jsonl.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            record = json.loads(line)
            train_examples.append(
                InputExample(texts=[record["query"], record["doc"]], label=float(record["label"]))
            )

            if idx % 100_000 == 0:
                log(f"Loaded examples: {idx:,}")

            if max_train_samples is not None and idx >= max_train_samples:
                log(f"Reached max_train_samples={max_train_samples:,}, stopping load early.")
                break

    if not train_examples:
        raise RuntimeError("No training examples were loaded. Cannot train reranker.")

    log(f"Total loaded examples: {len(train_examples):,}")
    log(f"Initializing CrossEncoder model: {model_name}")

    model = CrossEncoder(
        model_name,
        num_labels=1,
        max_length=512,
    )

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    warmup_steps = max(100, int(len(train_dataloader) * epochs * 0.1))

    model_dir = run_dir / "reranker_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"Starting training | epochs={epochs}, batch_size={batch_size}, batches_per_epoch={len(train_dataloader):,}, warmup_steps={warmup_steps:,}"
    )

    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=str(model_dir),
    )

    result = {
        "run_id": run_id,
        "model_dir": str(model_dir),
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_examples": len(train_examples),
    }

    result_path = run_dir / "train_stats.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    artifacts_volume.commit()
    log(f"Training complete. Model saved to {model_dir}")
    log(f"Training stats saved to {result_path}")

    return result


@app.local_entrypoint()
def main(
    run_id: str = "",
    negatives_per_query: int = 5,
    bm25_top_k: int = 50,
    log_every: int = 50,
    epochs: int = 3,
    batch_size: int = 128,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_train_samples: int = 0,
    num_workers: int = 32,
    chunk_size: int = 200,
    candidate_pool_multiplier: int = 32,
):
    actual_run_id = run_id or datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")

    print(f"[{_now()}] Starting run_id={actual_run_id}", flush=True)
    print(f"[{_now()}] Syncing local CSV files to Modal data volume...", flush=True)

    required_files = [
        "laws_de.csv",
        "court_considerations.csv",
        "train.csv",
    ]

    with data_volume.batch_upload(force=True) as batch:
        for filename in required_files:
            local_path = LOCAL_DATA_DIR / filename
            if not local_path.exists():
                raise FileNotFoundError(f"Required file not found: {local_path}")
            batch.put_file(str(local_path), filename)
            print(f"[{_now()}] Uploaded {local_path} -> /data/{filename}", flush=True)

    build_config = {
        "run_id": actual_run_id,
        "negatives_per_query": negatives_per_query,
        "bm25_top_k": bm25_top_k,
        "log_every": log_every,
        "num_workers": num_workers,
        "chunk_size": chunk_size,
        "candidate_pool_multiplier": candidate_pool_multiplier,
    }

    train_config = {
        "run_id": actual_run_id,
        "epochs": epochs,
        "batch_size": batch_size,
        "model_name": model_name,
        "max_train_samples": max_train_samples if max_train_samples > 0 else None,
    }

    build_result = build_training_data_remote.remote(build_config)
    print(f"[{_now()}] Build result: {json.dumps(build_result, indent=2)}", flush=True)

    train_result = train_reranker_remote.remote(train_config)
    print(f"[{_now()}] Train result: {json.dumps(train_result, indent=2)}", flush=True)

    print("\nDownload model to local machine with:")
    print(
        f"modal volume get {VOLUME_NAME} {actual_run_id}/reranker_model ./models/{actual_run_id}_reranker_model"
    )
    print("Download run stats with:")
    print(f"modal volume get {VOLUME_NAME} {actual_run_id} ./output/modal_runs/{actual_run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full BM25 + training data build + B200 reranker training on Modal."
    )
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--negatives-per-query", type=int, default=5)
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--candidate-pool-multiplier", type=int, default=32)
    args = parser.parse_args()

    # argparse path uses local_entrypoint style for standard python execution.
    main(
        run_id=args.run_id,
        negatives_per_query=args.negatives_per_query,
        bm25_top_k=args.bm25_top_k,
        log_every=args.log_every,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model_name,
        max_train_samples=args.max_train_samples,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        candidate_pool_multiplier=args.candidate_pool_multiplier,
    )
