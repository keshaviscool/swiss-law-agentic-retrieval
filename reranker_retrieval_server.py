"""
retrieval_server.py
===================
Modal-hosted retrieval API.

Architecture
------------
  1. FAISS IVF (BGE-M3 dense)  →  top-200 candidates   [fast, ~20 GB on volume]
  2. ColBERT v2 MaxSim re-rank →  top-k final results   [precise, ~500 MB model]
  3. Exposed as a single HTTPS web endpoint

Deploy:
    modal deploy retrieval_server.py

Call from notebook:
    POST https://<your-modal-url>--retrieve.modal.run
    Body: {"query": "...", "k": 10, "candidate_k": 200}
"""

import modal

# ---------------------------------------------------------------------------
# Infra
# ---------------------------------------------------------------------------
FAISS_INDEX_PATH  = "/data/faiss_index"
COLBERT_MODEL     = "colbert-ir/colbertv2.0"
BGE_MODEL         = "BAAI/bge-m3"
DEFAULT_CANDIDATE_K = 200   # FAISS retrieves this many, ColBERT re-ranks
DEFAULT_FINAL_K     = 15    # how many to return to the caller

app    = modal.App("swiss-law-retrieval")
volume = modal.Volume.from_name("swiss-law-volume", create_if_missing=False)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "faiss-cpu",
        "sentence-transformers",
        "ragatouille",
        "langchain",
        "langchain-community",
        "numpy",
        "fastapi",
        "pydantic",
    )
)

# ---------------------------------------------------------------------------
# Retrieval class — loaded once per container, reused across requests
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu="T4",               # T4 (16 GB) is enough for ColBERT re-ranking
    memory=32768,           # 32 GB RAM for the FAISS index + docstore
    timeout=300,
    volumes={"/data": volume},
    keep_warm=1,            # keep one container alive → no cold-start latency
)
class Retriever:

    @modal.enter()
    def load(self):
        """
        Runs once when the container starts (not on every request).
        Loads FAISS index and ColBERT model into memory.
        """
        import time
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragatouille import RAGPretrainedModel

        print("Loading BGE-M3 embeddings...")
        t0 = time.time()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=BGE_MODEL,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print("Loading FAISS index...")
        self.vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        n = self.vectorstore.index.ntotal
        print(f"  FAISS loaded: {n:,} vectors  ({time.time()-t0:.1f}s)")

        print("Loading ColBERT v2 model...")
        t1 = time.time()
        # RAGatouille in rerank-only mode — we don't build a ColBERT index,
        # we just use the model for MaxSim scoring on candidates.
        self.colbert = RAGPretrainedModel.from_pretrained(COLBERT_MODEL)
        print(f"  ColBERT loaded  ({time.time()-t1:.1f}s)")
        print("Retriever ready ✅")

    @modal.method()
    def retrieve(
        self,
        query: str,
        final_k: int = DEFAULT_FINAL_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ) -> list[dict]:
        """
        Two-stage retrieval:
          Stage 1 — FAISS IVF ANN → top candidate_k docs
          Stage 2 — ColBERT MaxSim re-rank → top final_k docs

        Returns list of dicts:
          {"citation": str, "text": str, "score": float, "source": str}
        """
        import time

        # ── Stage 1: Dense FAISS retrieval ────────────────────────────────────
        t0 = time.time()
        candidate_docs = self.vectorstore.similarity_search_with_score(
            query, k=candidate_k
        )
        t_faiss = time.time() - t0
        print(f"FAISS: {len(candidate_docs)} candidates in {t_faiss*1000:.0f}ms")

        if not candidate_docs:
            return []

        # Extract texts and metadata
        texts     = [doc.page_content for doc, _ in candidate_docs]
        metadatas = [doc.metadata     for doc, _ in candidate_docs]

        # ── Stage 2: ColBERT re-ranking ────────────────────────────────────────
        t1 = time.time()
        try:
            reranked = self.colbert.rerank(
                query=query,
                documents=texts,
                k=min(final_k, len(texts)),
            )
            t_colbert = time.time() - t1
            print(f"ColBERT: reranked to {len(reranked)} in {t_colbert*1000:.0f}ms")

            # RAGatouille returns list of dicts: {"content": str, "score": float, ...}
            results = []
            for r in reranked:
                # find original metadata by matching content
                content = r.get("content", "")
                meta    = {}
                for i, t in enumerate(texts):
                    if t == content:
                        meta = metadatas[i]
                        break

                results.append({
                    "citation": meta.get("citation", ""),
                    "text":     content,
                    "score":    float(r.get("score", 0.0)),
                    "source":   meta.get("source", ""),
                })

        except Exception as e:
            # ColBERT fallback: return FAISS top-k with dense scores
            print(f"ColBERT failed ({e}), falling back to FAISS scores")
            results = []
            for doc, score in candidate_docs[:final_k]:
                meta = doc.metadata
                results.append({
                    "citation": meta.get("citation", ""),
                    "text":     doc.page_content,
                    "score":    float(score),
                    "source":   meta.get("source", ""),
                })

        return results


# ---------------------------------------------------------------------------
# Web endpoint — exposes Retriever.retrieve() as HTTPS POST
# ---------------------------------------------------------------------------
from pydantic import BaseModel

class RetrieveRequest(BaseModel):
    query:       str
    k:           int = DEFAULT_FINAL_K
    candidate_k: int = DEFAULT_CANDIDATE_K

class RetrieveResponse(BaseModel):
    results: list[dict]
    query:   str
    timing:  dict

@app.function(
    image=image,
    timeout=120,
    min_containers=1,
)
@modal.web_endpoint(method="POST", label="retrieve")
def retrieve_endpoint(req: RetrieveRequest) -> RetrieveResponse:
    import time
    t0 = time.time()

    retriever = Retriever()
    results   = retriever.retrieve.remote(
        query=req.query,
        final_k=req.k,
        candidate_k=req.candidate_k,
    )

    elapsed = time.time() - t0
    return RetrieveResponse(
        results=results,
        query=req.query,
        timing={"total_ms": round(elapsed * 1000)},
    )


# ---------------------------------------------------------------------------
# Local test entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def test():
    retriever = Retriever()
    results   = retriever.retrieve.remote(
        "Schadensersatz bei Vertragsverletzung ZGB",
        final_k=5,
        candidate_k=50,
    )
    for r in results:
        print(f"[{r['score']:.3f}] {r['citation']}  |  {r['text'][:80]}...")