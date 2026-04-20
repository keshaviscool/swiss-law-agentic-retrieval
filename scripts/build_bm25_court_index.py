import modal

app = modal.App("build-bm25-court-index")

# Volume for output
volume = modal.Volume.from_name("bm25-court-index", create_if_missing=True)
VOLUME_PATH = "/output"
OUTPUT_FILE = f"{VOLUME_PATH}/bm25_court_index.pkl"

# Image with local CSV baked in
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "rank_bm25",
        "pandas",
        "numpy",
        "tqdm",
    )
    .add_local_file(
        "../data/court_considerations.csv",
        remote_path="/data/court.csv",
    )
)

@app.function(
    image=image,
    cpu=32,
    memory=128 * 1024,
    timeout=60 * 60,
    volumes={VOLUME_PATH: volume},
)
def build_index():
    import pickle
    import pandas as pd
    from tqdm import tqdm
    from rank_bm25 import BM25Okapi

    print("Reading CSV in chunks...")

    tokenized_corpus = []
    citations = []

    CHUNK_SIZE = 100_000

    def tokenize(text: str):
        return text.lower().split()

    total_rows = 0

    for chunk in pd.read_csv("/data/court.csv", chunksize=CHUNK_SIZE):
        texts = chunk["text"].fillna("").astype(str).tolist()
        cites = chunk["citation"].astype(str).tolist()

        tokenized_chunk = [tokenize(t) for t in texts]

        tokenized_corpus.extend(tokenized_chunk)
        citations.extend(cites)

        total_rows += len(chunk)
        print(f"Processed {total_rows:,} rows")

    print(f"Total documents: {len(tokenized_corpus):,}")
    print("Building BM25 index...")

    bm25 = BM25Okapi(tokenized_corpus)

    print("Index built. Saving...")

    payload = {
        "bm25": bm25,
        "citations": citations,
        "texts": tokenized_corpus,  # optional but useful for reranking
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(payload, f, protocol=5)

    print("Saved successfully.")

    volume.commit()
    print("Committed to volume.")


@app.local_entrypoint()
def main():
    print("Starting BM25 index build on Modal...")
    build_index.remote()
    print("Done.")