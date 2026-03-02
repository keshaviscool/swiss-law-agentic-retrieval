import modal

app = modal.App("swiss-law-faiss-builder")

volume = modal.Volume.from_name("swiss-law-volume", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "langchain",
        "langchain-community",
        "langchain-text-splitters",
        "sentence-transformers",
        "faiss-cpu",
        "pandas",
        "numpy",
        "tqdm"
    )
)

@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 6,
    volumes={"/data": volume},
)
def build_faiss_index():
    import time
    import numpy as np
    from langchain_community.document_loaders import CSVLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.docstore.document import Document

    print("Loading CSV files...")

    loader = CSVLoader(file_path="/data/laws_de.csv")
    laws_de = loader.load()

    loader = CSVLoader(file_path="/data/court_considerations.csv")
    court_considerations = loader.load()

    print("Splitting documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    laws_de = text_splitter.split_documents(laws_de)
    court_considerations = text_splitter.split_documents(court_considerations)

    all_docs = laws_de + court_considerations
    total = len(all_docs)

    print(f"Total chunks after splitting: {total}")
    print("Loading embedding model...")

    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"batch_size": 512}
    )

    print("Embedding documents...")

    BATCH_SIZE = 512
    all_embeddings = []
    start_time = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch_docs = all_docs[batch_start: batch_start + BATCH_SIZE]
        batch_texts = [doc.page_content for doc in batch_docs]

        batch_embeddings = embedder.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)

        done = min(batch_start + BATCH_SIZE, total)
        remaining = total - done
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = remaining / rate if rate > 0 else 0

        print(
            f"  Embedded {done}/{total} chunks "
            f"({done / total * 100:.1f}%) | "
            f"Remaining: {remaining} | "
            f"Rate: {rate:.0f} chunks/s | "
            f"ETA: {eta:.0f}s"
        )

    print(f"Embedding complete in {time.time() - start_time:.1f}s. Building FAISS index...")

    vectorstore = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, all_embeddings)],
        embedding=embedder,
        metadatas=[doc.metadata for doc in all_docs]
    )

    print("Saving FAISS index...")

    vectorstore.save_local("/data/faiss_index")

    print("Done.")
    return "FAISS index built successfully!"
