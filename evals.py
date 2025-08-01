import torch
import dualcodec
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import argparse, csv, pathlib
import matplotlib.pyplot as plt

from metrics import sisdr, multiscale_stft_loss


def load_audio(path: str, required_sr: int = 24000) -> torch.Tensor:
    sr, data = wavfile.read(path)
    if sr != required_sr:
        gcd = np.gcd(sr, required_sr)
        data = resample_poly(data, required_sr // gcd, sr // gcd)
    return torch.from_numpy(data)


def load_model():
    model_id_base = "12hz_v1"
    dualcodec_model = dualcodec.get_model(model_id_base)
    dualcodec_inference_model = dualcodec.Inference(dualcodec_model=dualcodec_model)
    return dualcodec_inference_model


def evaluate(model: torch.nn.Module, samples: list[torch.Tensor], device: str = "cpu") -> dict[str, float]:
    sisdr_scores, STFT_losses= [], []
    
    with torch.no_grad():
        for audio in samples:
            audio = audio.to(device)
            audio = audio.reshape(1, 1, -1)

            # reconstruction
            semantic_codes, acoustic_codes = model.encode(audio, n_quantizers=8)
            recon = model.decode(semantic_codes, acoustic_codes)
            
            # squeeze and align
            audio = audio.squeeze(0).squeeze(0).cpu()
            recon = recon.squeeze(0).squeeze(0).cpu()
            length = min(audio.shape[-1], recon.shape[-1])
            audio, recon = audio[:length], recon[:length]
            
            sisdr_scores.append(sisdr(recon, audio).item())
            STFT_losses.append(multiscale_stft_loss(recon, audio).item())

    return {
        "sisdr": float(np.mean(sisdr_scores)),
        "STFT_loss": float(np.mean(STFT_losses)),
    }
    
    
SAMPLES_DIR = "/home/vansh/dualcodec-exp/audio_samples"
OUTPUT_FILE = "results.csv"


def main():
    wav_paths = sorted(str(p) for p in pathlib.Path(SAMPLES_DIR).glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError("No .wav found in " + SAMPLES_DIR)
    samples = [load_audio(p).float() for p in wav_paths]
    
    print(f"Loaded {len(samples)} wavs from {SAMPLES_DIR}")
    
    MODELS = [load_model()]
    
    fieldnames = ["model", "sisdr", "STFT_loss"]
    rows: list[dict[str, float | str]] = []

    for model in MODELS:
        stats = evaluate(model, samples)                                              
        rows.append({"model": "dualcodec", **stats})
        print(f"Model: {model}, SISDRI: {stats['sisdr']}, STFT_loss: {stats['STFT_loss']}")

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = [m for m in fieldnames if m != "model"]
    for m in metrics:
        plt.figure(figsize=(6, 3))
        plt.title(m)
        plt.bar([r["model"] for r in rows], [r[m] for r in rows])
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = f"{m}.png"
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"Plot saved → {fname}")


if __name__ == "__main__":
    main()

