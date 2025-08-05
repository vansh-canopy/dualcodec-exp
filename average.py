import re
from pathlib import Path
from safetensors.torch import load_file, save_file


CHECKPOINT_DIR  = "/home/vansh/dualcodec-vansh/output_checkpoints/dualcodec_stream_with_whisper/checkpoint/" 
OUTPUT_DIR = "/home/vansh/dualcodec-vansh/averaged_models/"   
PREFIX = "epoch-0000_step-"  
DEVICE = "cpu"

LAST_K_STEPS = 20 
DECAY_RATES = [0.95, 0.9, 0.85]

# Only consider checkpoints whose training step falls within this range (inclusive)
MIN_STEP = 210000         # lower bound; change as needed
MAX_STEP = 260000         # upper bound; change as needed                       

ROOT = Path(CHECKPOINT_DIR).expanduser().resolve()
STEP_REGEX = re.compile(r"step-([0-9]*\.?[0-9]+)")  


def find_checkpoints(root: Path, prefix: str):
    candidates = []

    for ckp in root.rglob("model.safetensors"):
        folder = ckp.parent.name    
        if not folder.startswith(prefix):
            continue
        m = STEP_REGEX.search(folder)
        if m:
            step_val = float(m.group(1))
            if not (MIN_STEP <= step_val <= MAX_STEP):
                continue
            candidates.append((step_val, ckp))
    
    return sorted(candidates, key=lambda x: -x[0])


def ema_average(av, new, decay_factor):
    # av = β·av + (1-β)·new
    if not av:
        return {k: v.clone() for k, v in new.items()}
    for k, v in new.items():
        av[k].mul_(decay_factor).add_(v, alpha=1.0 - decay_factor)
    return av


def main():
    checkpoints = find_checkpoints(ROOT, PREFIX)
    chosen = checkpoints[:LAST_K_STEPS] if LAST_K_STEPS else checkpoints
    print(f"Found {len(checkpoints)} ckpts, using {len(chosen)} by recency")
    
    for DECAY in DECAY_RATES:
        out_file = f"{OUTPUT_DIR}/averaged_model_step_0{MAX_STEP}_decay_{DECAY}.safetensors"
        aver_state = {}
        for _, path in chosen:
            sd = load_file(path, device=DEVICE)
            aver_state = ema_average(aver_state, sd, DECAY)

        save_file(aver_state, out_file)
        print(f"Saved {out_file}")
    

if __name__ == "__main__":
    main()
    