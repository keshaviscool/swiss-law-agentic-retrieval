import modal
import os
import tempfile

VOLUME_NAME = "reranker-model-v2"
REMOTE_DIR  = "/reranker_bge_swiss_law_b200"
HF_REPO     = "keshavsharma/reranker-bge-swiss-law"  # change to your HF username

app = modal.App("push-reranker-to-hf")

volume = modal.Volume.from_name(VOLUME_NAME)

@app.function(
    image=modal.Image.debian_slim().pip_install("huggingface_hub"),
    volumes={"/model": volume},
    secrets=[modal.Secret.from_name("hfsecret")],
)
def push_to_hub():
    from huggingface_hub import HfApi
    import os

    api = HfApi(token=os.environ["HF_TOKEN"])

    # Create repo if it doesn't exist
    api.create_repo(HF_REPO, exist_ok=True, private=True)

    print(f"Uploading from /model{REMOTE_DIR} to {HF_REPO}...")
    api.upload_folder(
        folder_path=f"/model{REMOTE_DIR}",
        repo_id=HF_REPO,
        repo_type="model",
    )
    print(f"Done. Model live at: https://huggingface.co/{HF_REPO}")

@app.local_entrypoint()
def main():
    push_to_hub.remote()