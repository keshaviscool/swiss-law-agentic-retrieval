import modal
import os

app = modal.App("reranker-training-b200-final")

samples_volume = modal.Volume.from_name("reranker-samples-v2")
model_volume   = modal.Volume.from_name("reranker-model-v2", create_if_missing=True)

SAMPLES_PATH = "/data"
MODEL_PATH   = "/model"
SAMPLES_FILE = f"{SAMPLES_PATH}/reranker_samples_v2.json"
OUTPUT_DIR   = f"{MODEL_PATH}/reranker_bge_swiss_law_b200"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "sentence-transformers==3.0.1",
        "transformers==4.44.2",
        "scikit-learn",
        "scipy",
        "tqdm",
        "accelerate",
        "numpy",
    )
)


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 120,
    volumes={
        SAMPLES_PATH: samples_volume,
        MODEL_PATH:   model_volume,
    },
)
def train_reranker():
    import json
    import random
    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from torch.utils.data import DataLoader

    # ── B200 performance flags ────────────────────────────────────────────────
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load samples ──────────────────────────────────────────────────────────
    print(f"Loading samples from {SAMPLES_FILE}...")
    with open(SAMPLES_FILE) as f:
        saved = json.load(f)

    raw_samples = saved["samples"]
    pos = sum(1 for s in raw_samples if s["label"] == 1.0)
    neg = sum(1 for s in raw_samples if s["label"] == 0.0)
    print(f"Total: {len(raw_samples)} | Pos: {pos} | Neg: {neg} | Ratio: {neg/pos:.1f}:1")

    samples = [
        InputExample(texts=s["texts"], label=float(s["label"]))
        for s in raw_samples
    ]

    random.seed(42)
    random.shuffle(samples)

    split       = int(0.9 * len(samples))
    train_split = samples[:split]
    eval_split  = samples[split:]
    print(f"Train: {len(train_split)} | Eval: {len(eval_split)}")

    # ── Load model ────────────────────────────────────────────────────────────
    # Strategy: load in float32, then cast only the heavy encoder layers to
    # bfloat16. The classifier head stays in float32, so scores come out as
    # float32 natively — CEBinaryClassificationEvaluator never sees bfloat16
    # tensors and the numpy() call works without any monkey-patching.
    print("Loading BAAI/bge-reranker-v2-m3...")
    model = CrossEncoder(
        "BAAI/bge-reranker-v2-m3",
        num_labels=1,
        max_length=512,
        device="cuda",
        # load in float32 intentionally — we cast below
    )

    # Cast encoder to bf16 for speed, keep classifier in float32 for stability
 # Cast everything except the classifier to bf16
    # for name, param in model.model.named_parameters():
    #     if "classifier" not in name:
    #         param.data = param.data.to(torch.bfloat16)
    # print("All non-classifier params cast to bfloat16.")

    # torch.compile — gives ~15-25% throughput boost on B200
    # try:
    #     model.model = torch.compile(model.model, mode="reduce-overhead")
    #     print("torch.compile applied.")
    # except Exception as e:
    #     print(f"torch.compile skipped: {e}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    # B200 has 192 GB HBM3e — batch 128 uses ~50 GB, well within limits.
    # Larger batches = more implicit negatives per step = better ranking.
    BATCH_SIZE = 64

    train_dataloader = DataLoader(
        train_split,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        eval_split,
        name="swiss-law-bge-eval",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    EPOCHS       = 3
    warmup_steps = int(0.1 * len(train_dataloader) * EPOCHS)

    print(f"Training | epochs={EPOCHS} | batch={BATCH_SIZE} | warmup={warmup_steps}")
    print(f"Steps per epoch: {len(train_dataloader)}")

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=EPOCHS,
        optimizer_params={"lr": 1.5e-5},
        warmup_steps=warmup_steps,
        output_path=OUTPUT_DIR,
        save_best_model=True,       # saves whichever epoch had best eval AP
        evaluation_steps=100,       # eval every 100 steps, not just end of epoch
        show_progress_bar=True,
        use_amp=False,              # we handle precision manually above
    )

    model_volume.commit()
    print(f"\nModel saved to {OUTPUT_DIR}")

    # ── Sanity check ──────────────────────────────────────────────────────────
    pos_ex = next(s for s in eval_split if s.label == 1.0)
    neg_ex = next(s for s in eval_split if s.label == 0.0)
    scores = model.predict([pos_ex.texts, neg_ex.texts])

    print(f"\nSanity check (eval set examples):")
    print(f"  Positive pair score : {scores[0]:.4f}  (label=1.0)")
    print(f"  Negative pair score : {scores[1]:.4f}  (label=0.0)")
    print(f"  Separation          : {scores[0] - scores[1]:.4f}  (higher = better)")


@app.local_entrypoint()
def main():
    print("Submitting BGE reranker training to Modal B200...")
    train_reranker.remote()
    print("Done. Monitor: modal app logs reranker-training-b200-final")