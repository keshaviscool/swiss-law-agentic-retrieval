"""
Download reranker_samples.json (and errors) from the Modal volume.

Run:  python download_reranker_samples.py
"""
import modal
import os

LOCAL_DIR   = "./reranker_data"
VOLUME_NAME = "reranker-samples"

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    vol = modal.Volume.from_name(VOLUME_NAME)

    print("Files in volume:")
    for entry in vol.listdir("/"):
        print(f"  {entry.path}  ({entry.stat().st_size:,} bytes)")

    for fname in ["reranker_samples.json", "reranker_errors.json"]:
        local_path = os.path.join(LOCAL_DIR, fname)
        print(f"Downloading {fname} -> {local_path} ...")
        with open(local_path, "wb") as out:
            for chunk in vol.read_file(f"/{fname}"):
                out.write(chunk)
        print(f"  Done ({os.path.getsize(local_path):,} bytes)")

    print(f"\nAll files saved to {os.path.abspath(LOCAL_DIR)}/")

if __name__ == "__main__":
    main()