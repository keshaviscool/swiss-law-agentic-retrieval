import modal
import os

app = modal.App("reranker-training-v2")

# Read from v2 samples volume, write model to new volume
samples_volume = modal.Volume.from_name("reranker-samples-v2")
model_volume   = modal.Volume.from_name("reranker-model-v2", create_if_missing=True)

SAMPLES_PATH      = "/data"
MODEL_PATH        = "/model"
SAMPLES_FILE      = f"{SAMPLES_PATH}/reranker_samples_v2.json"
OUTPUT_DIR        = f"{MODEL_PATH}/reranker_bge_swiss_law"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "sentence-transformers==3.0.1",
        "transformers==4.44.2",
        "scikit-learn",
        "scipy",
        "tqdm",
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 120,
    volumes={
        SAMPLES_PATH: samples_volume,
        MODEL_PATH:   model_volume,
    },
)
def train_reranker():
    import json
    import random
    from sentence_transformers import CrossEncoder, InputExample
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from torch.utils.data import DataLoader

    # ── Load samples ──────────────────────────────────────────────────────────
    print(f"Loading samples from {SAMPLES_FILE}...")
    with open(SAMPLES_FILE) as f:
        saved = json.load(f)

    raw_samples = saved["samples"]
    pos = sum(1 for s in raw_samples if s["label"] == 1.0)
    neg = sum(1 for s in raw_samples if s["label"] == 0.0)
    print(f"Total: {len(raw_samples)} | Positives: {pos} | Negatives: {neg} | Ratio: {neg/pos:.1f}:1")

    samples = [
        InputExample(texts=s["texts"], label=float(s["label"]))
        for s in raw_samples
    ]

    # ── Split ─────────────────────────────────────────────────────────────────
    random.seed(42)
    random.shuffle(samples)
    split       = int(0.9 * len(samples))
    train_split = samples[:split]
    eval_split  = samples[split:]
    print(f"Train: {len(train_split)} | Eval: {len(eval_split)}")

    # ── Load BGE reranker — much better than MiniLM for German legal text ─────
    print("Loading BAAI/bge-reranker-v2-m3...")
    model = CrossEncoder(
        "BAAI/bge-reranker-v2-m3",
        num_labels=1,
        max_length=512,
        device="cuda",
    )

    # ── DataLoader ────────────────────────────────────────────────────────────
    # Batch 32 for BGE (larger than MiniLM, use slightly smaller batch)
    train_dataloader = DataLoader(train_split, shuffle=True, batch_size=32)

    # ── Evaluator ─────────────────────────────────────────────────────────────
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        eval_split,
        name="swiss-law-bge-eval",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    warmup_steps = int(0.1 * len(train_dataloader) * 10)
    print(f"Training | epochs=10 | batch=32 | warmup={warmup_steps}")

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=3,
        learning_rate=2e-5,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_DIR,
        save_best_model=True,
        evaluation_steps=100,
        show_progress_bar=True,
    )

    model_volume.commit()
    print(f"Model saved to {OUTPUT_DIR}")

    # ── Quick sanity check ────────────────────────────────────────────────────
    pos_ex = next(s for s in train_split if s.label == 1.0)
    neg_ex = next(s for s in train_split if s.label == 0.0)
    scores = model.predict([pos_ex.texts, neg_ex.texts])
    print(f"\nSanity check:")
    print(f"  Positive pair score: {scores[0]:.4f}  (label=1.0)")
    print(f"  Negative pair score: {scores[1]:.4f}  (label=0.0)")
    print(f"  Separation: {scores[0] - scores[1]:.4f}  (higher is better)")


@app.local_entrypoint()
def main():
    print("Submitting BGE reranker training to Modal H100...")
    train_reranker.remote()
    print("Done. Monitor: modal app logs reranker-training-v2")