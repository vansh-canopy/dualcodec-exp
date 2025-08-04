from pathlib import Path
import re
from huggingface_hub import create_repo
import dualcodec

MODELS_DIR = Path("/home/vansh/dualcodec-exp/averaged_models")
STEP_REGEX = re.compile(r"step[_-](\d{7})")

HF_USERNAME = "vanshjjw"

def main(do_push=True):
    if not do_push:
        return
    
    for filename in sorted(MODELS_DIR.glob("*.safetensors")):
        m = STEP_REGEX.search(filename.name)          
        step_millions = int(m.group(1)) / 1e6  
            
        repo_name = f"vansh-dualcodec-step-{step_millions:.3f}"
        repo_name_complete = f"{HF_USERNAME}/{repo_name}"
        create_repo(repo_name_complete, private=True, exist_ok=True)

        # ensure the model loads correctly before pushing
        dualcodec_model = dualcodec.get_model("12hz_v3", str(MODELS_DIR), name=filename.name)
        dualcodec_model.push_to_hub(repo_name_complete)
    

if __name__ == "__main__":
    main(do_push=True)