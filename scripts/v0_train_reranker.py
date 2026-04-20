import modal
import os

# ── Modal setup ───────────────────────────────────────────────────────────────
app = modal.App("reranker-training")

# Same volume where reranker_samples.json was saved
samples_volume = modal.Volume.from_name("reranker-samples")

# New volume to persist the trained reranker
model_volume = modal.Volume.from_name("reranker-model-crossmini-epoch-8", create_if_missing=True)

SAMPLES_VOLUME_PATH = "/data"
MODEL_VOLUME_PATH   = "/model"
SAVE_PATH           = f"{SAMPLES_VOLUME_PATH}/reranker_samples.json"
OUTPUT_PATH         = f"{MODEL_VOLUME_PATH}/reranker_swiss_law"

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
    timeout=60 * 120,   # 2 hours
    volumes={
        SAMPLES_VOLUME_PATH: samples_volume,
        MODEL_VOLUME_PATH:   model_volume,
    },
)
def train_reranker():
    import json
    import random
    from sentence_transformers import CrossEncoder, InputExample
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from torch.utils.data import DataLoader

    # ── Load samples from volume ──────────────────────────────────────────────
    print(f"Loading samples from {SAVE_PATH} ...")
    with open(SAVE_PATH) as f:
        saved = json.load(f)

    raw_samples = saved["samples"]
    print(f"Total raw samples: {len(raw_samples)}")

    pos = sum(1 for s in raw_samples if s["label"] == 1.0)
    neg = sum(1 for s in raw_samples if s["label"] == 0.0)
    print(f"Positives: {pos} | Negatives: {neg} | Ratio: {neg/pos:.1f}:1")

    # ── Convert to InputExamples ──────────────────────────────────────────────
    reranker_samples = [
        InputExample(texts=s["texts"], label=float(s["label"]))
        for s in raw_samples
    ]

    # ── Train / eval split ────────────────────────────────────────────────────
    random.seed(42)
    random.shuffle(reranker_samples)
    split      = int(0.9 * len(reranker_samples))
    train_split = reranker_samples[:split]
    eval_split  = reranker_samples[split:]
    print(f"Train: {len(train_split)} | Eval: {len(eval_split)}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading CrossEncoder base model...")
    model = CrossEncoder(
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        num_labels=1,
        max_length=512,
        device="cuda",
    )

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_dataloader = DataLoader(
        train_split,
        shuffle=True,
        batch_size=64,   # H100 has 80GB — push batch size up from 32
    )

    # ── Evaluator ─────────────────────────────────────────────────────────────
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        eval_split,
        name="swiss-law-reranker-eval",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    warmup_steps = int(0.1 * len(train_dataloader) * 3)
    print(f"Starting training | epochs=3 | warmup_steps={warmup_steps} | batch=64")

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=8,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_PATH,
        save_best_model=True,
        show_progress_bar=True,
    )

    model_volume.commit()
    print(f"Reranker saved to {OUTPUT_PATH}")

    # ── Quick sanity check ────────────────────────────────────────────────────
    print("\nSanity check — scoring 2 sample pairs...")
    sample_pos = train_split[0]
    sample_neg = next(s for s in train_split if s.label == 0.0)

    scores = model.predict([sample_pos.texts, sample_neg.texts])
    print(f"  Positive pair score : {scores[0]:.4f}  (label={sample_pos.label})")
    print(f"  Negative pair score : {scores[1]:.4f}  (label={sample_neg.label})")


@app.local_entrypoint()
def main():
    print("Submitting reranker training job to Modal H100...")
    train_reranker.remote()
    print("Done. Monitor with: modal app logs reranker-training")