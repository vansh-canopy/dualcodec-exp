import re
from pathlib import Path
from safetensors.torch import load_file, save_file


CHECKPOINT_DIR  = "/home/vansh/dualcodec-exp/output_checkpoints_1/dualcodec_25hzv1_finetune_1/checkpoint/"

OUTPUT_DIR = "/home/vansh/dualcodec-exp/averaged_models/"   
PREFIX = "epoch"  
DEVICE = "cuda:3"

LAST_K_STEPS = 20
DECAY_RATES = [0.9, 0.95, 0.99]

# Only consider checkpoints whose training step falls within this range (inclusive)
MIN_STEP = 000000         # lower bound; change as needed
MAX_STEP = 130000         # upper bound; change as needed                       

ROOT = Path(CHECKPOINT_DIR).expanduser().resolve()
STEP_REGEX = re.compile(r"step-([0-9]*\.?[0-9]+)")  


def find_checkpoints(root: Path, prefix: str):
    candidates = []

    for ckp in root.rglob("model.safetensors"):
        print(f"ckp: {ckp}")
        folder = ckp.parent.name    
        if not folder.startswith(prefix):
            continue
        m = STEP_REGEX.search(folder)
        print(f"folder: {folder}, m: {m}")
        if m:
            step_val = float(m.group(1))
            if not (MIN_STEP <= step_val <= MAX_STEP):
                continue
            candidates.append((step_val, ckp))
    
    return sorted(candidates, key=lambda x: -x[0])


def is_decoder_key(key: str):
    low = key.lower()
    if "decoder" not in low:
        return False
    return True


def ema_average_decoder_only(av: dict, new: dict, decay_factor: float):
    if not av:
        return {k: v.clone() for k, v in new.items()}

    for k, v in new.items():
        if is_decoder_key(k):
            av[k].mul_(decay_factor).add_(v, alpha=1.0 - decay_factor)
    return av


def main():
    checkpoints = find_checkpoints(ROOT, PREFIX)
    chosen = checkpoints[:LAST_K_STEPS] if LAST_K_STEPS else checkpoints
    print(f"Found {len(checkpoints)} ckpts, using {len(chosen)} by recency")
    
    for DECAY in DECAY_RATES:
        out_file = f"{OUTPUT_DIR}/exp_1_step_{MAX_STEP}_last_{LAST_K_STEPS}_decay_{DECAY}.safetensors"
        # ensure output directory exists
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)

        aver_state = {}
        for _, path in chosen:
            sd = load_file(path, device=DEVICE)
            aver_state = ema_average_decoder_only(aver_state, sd, DECAY)

        save_file(aver_state, out_file)
        print(f"Saved {out_file}")
    

if __name__ == "__main__":
    main()
    