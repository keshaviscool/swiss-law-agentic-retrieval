"""
Run this locally after training completes:
    python download_adapter.py

Downloads the LoRA adapter from the Modal volume to ./adapter/
"""
import modal
import os
from modal.volume import FileEntryType

LOCAL_DIR = "./llama_adapter"
REMOTE_DIR = "/llama_swiss_law_adapter"


def _relative_remote_path(remote_root: str, remote_path: str) -> str:
    root = remote_root.strip("/")
    path = remote_path.strip("/")

    if path == root:
        return ""

    if path.startswith(root + "/"):
        return path[len(root) + 1 :]

    return os.path.basename(path)

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    volume = modal.Volume.from_name("llama-law-adapter-rank32-epoch3")

    # reload() is only valid inside running Modal functions; local scripts can skip it.
    if hasattr(volume, "reload"):
        try:
            volume.reload()
        except RuntimeError as e:
            if "can only be called from within a running function" not in str(e):
                raise

    print(f"Listing files in volume path: {REMOTE_DIR}")
    entries = volume.listdir(REMOTE_DIR, recursive=True)
    if not entries:
        raise RuntimeError(
            f"No files found under {REMOTE_DIR}. "
            "Make sure training finished and volume.commit() ran successfully."
        )

    for entry in entries:
        entry_type = str(entry.type).split(".")[-1]
        print(f"  {entry.path} [{entry_type}] size={entry.size}")

    files = [e for e in entries if e.type == FileEntryType.FILE]
    if not files:
        raise RuntimeError(f"No regular files found under {REMOTE_DIR}.")

    print(f"\nDownloading adapter to {LOCAL_DIR}/...")
    for i, entry in enumerate(files, start=1):
        rel_path = _relative_remote_path(REMOTE_DIR, entry.path)
        local_path = os.path.join(LOCAL_DIR, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with open(local_path, "wb") as f:
            bytes_written = volume.read_file_into_fileobj(entry.path, f)

        print(f"  [{i}/{len(files)}] {entry.path} -> {local_path} ({bytes_written} bytes)")

    print(f"Done. Adapter saved to {os.path.abspath(LOCAL_DIR)}")

if __name__ == "__main__":
    main()