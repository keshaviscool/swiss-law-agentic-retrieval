import argparse
import json
import pickle
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "swiss-law-cross-encoder-from-pkl"
ARTIFACT_VOLUME_NAME = "swiss-law-reranker-artifacts"
DATA_VOLUME_NAME = "swiss-law-input-data"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_DIR = PROJECT_ROOT / "data"
REMOTE_DATA_DIR = Path("/data")
REMOTE_ARTIFACT_DIR = Path("/artifacts")

app = modal.App(APP_NAME)
artifacts_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=3.0.0",
        "torch>=2.2.0",
        "scikit-learn>=1.3.0",
        "datasets>=2.14.0",
        "accelerate>=1.1.0",
    )
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@app.function(
    image=image,
    volumes={str(REMOTE_DATA_DIR): data_volume, str(REMOTE_ARTIFACT_DIR): artifacts_volume},
    gpu="B200",
    cpu=16,
    memory=196_608,
    timeout=60 * 60 * 8,
)
def train_cross_encoder_from_pickle_remote(config: dict) -> dict:
    import json
    import pickle
    import random
    from pathlib import Path

    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from torch.utils.data import DataLoader

    def log(msg: str) -> None:
        print(f"[{_now()}] [CROSS-ENCODER] {msg}", flush=True)

    run_id = config["run_id"]
    pkl_filename = config.get("pkl_filename", "cross_encoder_training_samples.pkl")
    model_name = config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    output_subdir = config.get("output_subdir", "cross_encoder_swiss_law_finetuned")

    epochs = int(config.get("epochs", 5))
    batch_size = int(config.get("batch_size", 256))
    train_ratio = float(config.get("train_ratio", 0.9))
    max_length = int(config.get("max_length", 512))
    seed = int(config.get("seed", 42))
    max_train_samples = int(config.get("max_train_samples", 0))

    run_dir = Path("/artifacts") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = Path("/data") / pkl_filename
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle file not found at {pkl_path}")

    log(f"Loading train samples from {pkl_path}")
    with pkl_path.open("rb") as f:
        train_samples = pickle.load(f)

    if not isinstance(train_samples, list) or len(train_samples) == 0:
        raise RuntimeError("Loaded pickle does not contain a non-empty list of training samples.")

    if max_train_samples > 0:
        train_samples = train_samples[:max_train_samples]
        log(f"Applied max_train_samples={max_train_samples:,}")

    rng = random.Random(seed)
    rng.shuffle(train_samples)

    if len(train_samples) < 2:
        raise RuntimeError("Need at least 2 samples for train/eval split.")

    split_idx = int(train_ratio * len(train_samples))
    split_idx = max(1, min(split_idx, len(train_samples) - 1))

    train_split = train_samples[:split_idx]
    eval_split = train_samples[split_idx:]

    log(f"Train samples: {len(train_split):,}")
    log(f"Eval samples : {len(eval_split):,}")

    model = CrossEncoder(
        model_name,
        num_labels=1,
        max_length=max_length,
    )

    train_dataloader = DataLoader(
        train_split,
        shuffle=True,
        batch_size=batch_size,
    )

    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        eval_split,
        name="swiss-law-eval",
    )

    warmup_steps = max(10, int(0.1 * len(train_dataloader) * epochs))
    output_path = run_dir / output_subdir

    log(
        "Starting fit "
        f"| epochs={epochs}, batch_size={batch_size}, batches_per_epoch={len(train_dataloader):,}, warmup_steps={warmup_steps:,}"
    )

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_path),
        save_best_model=True,
        show_progress_bar=True,
    )

    result = {
        "run_id": run_id,
        "pkl_path": str(pkl_path),
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_ratio": train_ratio,
        "max_length": max_length,
        "seed": seed,
        "num_samples_total": len(train_samples),
        "num_train_samples": len(train_split),
        "num_eval_samples": len(eval_split),
        "model_output_path": str(output_path),
    }

    stats_path = run_dir / "train_stats_from_pickle.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    artifacts_volume.commit()
    log(f"Training complete. Model saved to {output_path}")
    log(f"Stats saved to {stats_path}")

    return result


@app.local_entrypoint()
def main(
    run_id: str = "",
    pkl_filename: str = "cross_encoder_training_samples.pkl",
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 5,
    batch_size: int = 256,
    train_ratio: float = 0.9,
    max_length: int = 512,
    seed: int = 42,
    max_train_samples: int = 0,
    output_subdir: str = "cross_encoder_swiss_law_finetuned",
    auto_download: bool = True,
    local_download_dir: str = "./models",
):
    actual_run_id = run_id or datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")

    print(f"[{_now()}] Starting run_id={actual_run_id}", flush=True)

    local_pkl_path = LOCAL_DATA_DIR / pkl_filename
    if not local_pkl_path.exists():
        raise FileNotFoundError(
            f"Pickle file not found locally: {local_pkl_path}. "
            "Expected path from notebook save step."
        )

    print(f"[{_now()}] Uploading {local_pkl_path} to Modal data volume...", flush=True)
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_pkl_path), pkl_filename)

    train_config = {
        "run_id": actual_run_id,
        "pkl_filename": pkl_filename,
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_ratio": train_ratio,
        "max_length": max_length,
        "seed": seed,
        "max_train_samples": max_train_samples,
        "output_subdir": output_subdir,
    }

    result = train_cross_encoder_from_pickle_remote.remote(train_config)
    print(f"[{_now()}] Training result:\n{json.dumps(result, indent=2)}", flush=True)

    remote_model_path = f"{actual_run_id}/{output_subdir}"
    local_target = Path(local_download_dir) / f"{actual_run_id}_{output_subdir}"

    print("\nModel artifact is ready on Modal volume.")
    print(
        "Manual download command:\n"
        f"modal volume get {ARTIFACT_VOLUME_NAME} {remote_model_path} {local_target}"
    )

    if auto_download:
        local_target.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "modal",
            "volume",
            "get",
            ARTIFACT_VOLUME_NAME,
            remote_model_path,
            str(local_target),
        ]
        try:
            print(f"\n[{_now()}] Auto-downloading model to {local_target}...", flush=True)
            subprocess.run(cmd, check=True)
            print(f"[{_now()}] Download complete.", flush=True)
        except FileNotFoundError:
            print(
                "[WARN] modal CLI not found locally, so auto-download was skipped. "
                "Run the manual command above.",
                flush=True,
            )
        except subprocess.CalledProcessError as exc:
            print(
                "[WARN] Auto-download failed. Run the manual command above. "
                f"Details: {exc}",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Swiss-law CrossEncoder from prebuilt pickle samples on Modal B200 and optionally download artifacts."
    )
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--pkl-filename", type=str, default="cross_encoder_training_samples.pkl")
    parser.add_argument("--model-name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--output-subdir", type=str, default="cross_encoder_swiss_law_finetuned")
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument("--no-auto-download", dest="auto_download", action="store_false")
    parser.set_defaults(auto_download=True)
    parser.add_argument("--local-download-dir", type=str, default="./models")
    args = parser.parse_args()

    main(
        run_id=args.run_id,
        pkl_filename=args.pkl_filename,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        max_length=args.max_length,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        output_subdir=args.output_subdir,
        auto_download=args.auto_download,
        local_download_dir=args.local_download_dir,
    )
