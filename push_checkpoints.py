from pathlib import Path
import os
from huggingface_hub import HfApi, create_repo

FOLDER_TO_PUSH = Path("/home/vansh/dualcodec-exp/output_checkpoints").resolve()
HF_USERNAME = "vanshjjw"  # change if needed
REPO_NAME = f"dualcodec-baseline-checkpoints"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

api = HfApi()  # assumes you are already logged in via `huggingface-cli login`
create_repo(REPO_ID, private=True, exist_ok=True)
print(f"Created/using repo: {REPO_ID}")

api.upload_large_folder(repo_id=REPO_ID, folder_path=str(FOLDER_TO_PUSH), repo_type="model")
print("Upload complete.")