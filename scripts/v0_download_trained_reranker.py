"""
Download the trained reranker from the Modal volume.
Run:  python download_reranker_model.py
Then use:  CrossEncoder("./reranker_swiss_law")
"""
import modal
import os

LOCAL_DIR   = "./reranker_swiss_law"
VOLUME_NAME = "reranker-model-crossmini-epoch-8"
REMOTE_DIR  = "/reranker_swiss_law"

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    vol = modal.Volume.from_name(VOLUME_NAME)

    print("Files in volume:")
    entries = list(vol.listdir(REMOTE_DIR))
    # for entry in entries:
    #     print(f"  {entry.path}  ({entry.stat().st_size:,} bytes)")

    print(f"\nDownloading to {os.path.abspath(LOCAL_DIR)}/...")
    for entry in entries:
        fname      = os.path.basename(entry.path)
        local_path = os.path.join(LOCAL_DIR, fname)
        with open(local_path, "wb") as out:
            for chunk in vol.read_file(entry.path):
                out.write(chunk)
        print(f"  {fname} ({os.path.getsize(local_path):,} bytes)")

    print("\nDone. Load with:")
    print('  from sentence_transformers import CrossEncoder')
    print('  reranker = CrossEncoder("./reranker_swiss_law")')

if __name__ == "__main__":
    main()