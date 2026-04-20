"""
Download the v2 BGE reranker from Modal volume.
Run: python download_reranker_v2.py
"""
import modal
import os

LOCAL_DIR   = "./reranker_bge_swiss_law_b200"
VOLUME_NAME = "reranker-model-v2"
REMOTE_DIR  = "/reranker_bge_swiss_law_b200"

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    vol = modal.Volume.from_name(VOLUME_NAME)

    print("Files in volume:")
    entries = list(vol.listdir(REMOTE_DIR))
    # for e in entries:
    #     print(f"  {e.path}  ({e.stat().st_size:,} bytes)")

    print(f"\nDownloading to {os.path.abspath(LOCAL_DIR)}/...")
    for e in entries:
        fname = os.path.basename(e.path)
        local_path = os.path.join(LOCAL_DIR, fname)
        print("downloading", e.path, "...")
        with open(local_path, "wb") as out:
            for chunk in vol.read_file(e.path):
                out.write(chunk)
        print(f"  {fname} ({os.path.getsize(local_path):,} bytes)")

    print("\nDone. Load with:")
    print('  from sentence_transformers import CrossEncoder')
    print('  reranker = CrossEncoder("./reranker_bge_swiss_law")')

if __name__ == "__main__":
    main()