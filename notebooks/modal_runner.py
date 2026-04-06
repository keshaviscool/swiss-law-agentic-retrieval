import os
import modal
import pickle
from pathlib import Path

class ModalRunner:
    def __init__(self, app_name: str = "my_modal_app"):
        self.app = modal.App(app_name)
        self.download_dir = Path("../modal_downloads")
        self.download_dir.mkdir(exist_ok=True)

    def run(
            self,
            fn,
            input_data,
            gpu="T4",
            timeout=3600,
            image=None
    ):
        """
        fn: function to run remotely
        input_data: any pickleable object
        gpu: "T4", "A10G", etc.
        """
        if image is None:
            image = modal.Image.debian_slim().pip_install(
                "torch",
                "transformers",
                "sentence-transformers",
                "datasets",
                "faiss-cpu",
            )

        # serialize input
        input_bytes = pickle.dumps(input_data)

        @self.app.function(
            gpu=gpu,
            timeout=timeout,
            image=image,
        )
        def remote_fn(input_bytes):
            import pickle
            input_data = pickle.loads(input_bytes)
            print("🚀 Running on Modal GPU...")
            result = fn(input_data)

            print("✅ Done!")
            return result
        
        with self.app.run():
            result = remote_fn.call(input_bytes)

        # save locally
        output_path = self.download_dir / f"{fn.__name__}_output.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(result, f)
        print(f"📥 Result saved to {output_path}")

        return result
    