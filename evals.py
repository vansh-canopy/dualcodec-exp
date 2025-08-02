import torch
import torchaudio
import dualcodec
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import csv, pathlib
import matplotlib.pyplot as plt

from metrics import sisdr, multiscale_stft_loss, visqol_score


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


def evaluate(model: torch.nn.Module, samples: list[torch.Tensor], device: str = "cpu") -> dict[str, list[float]]:
    sisdr_scores, STFT_losses, visqol_scores = [], [], []
    
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
            
            # resample to 16000 for visqol
            audio = torchaudio.transforms.Resample(24000, 16000)(audio)
            recon = torchaudio.transforms.Resample(24000, 16000)(recon)
            
            visqol_scores.append(visqol_score(recon, audio))

    return {
        "sisdr": sisdr_scores,
        "STFT_loss": STFT_losses,
        "visqol": visqol_scores,
    }
    
    
SAMPLES_DIR = "/home/vansh/dualcodec-exp/audio_samples"
OUTPUT_FILE = "results.csv"


def main():
    wav_paths = sorted(str(p) for p in pathlib.Path(SAMPLES_DIR).glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError("No .wav found in " + SAMPLES_DIR)
    
    samples = []
    for p in wav_paths:
        sample, sr = torchaudio.load(p)
        if sr != 24000:
            sample = torchaudio.transforms.Resample(sr, 24000)(sample)
        samples.append(sample)
    
    print(f"Loaded {len(samples)} wavs from {SAMPLES_DIR}")
    
    MODELS = [("dualcodec", load_model())]

    fieldnames_summary = ["model", "sisdr", "STFT_loss", "visqol"]
    summary_rows: list[dict[str, float | str]] = []

    fieldnames_full = ["model", "sample_index", "sisdr", "STFT_loss", "visqol"]
    full_rows: list[dict[str, float | str]] = []

    all_results: dict[str, dict[str, list[float]]] = {}

    for model_name, model in MODELS:
        stats = evaluate(model, samples)
        all_results[model_name] = stats

        # compute means
        means = {metric: float(np.mean(values)) for metric, values in stats.items()}
        summary_rows.append({"model": model_name, **means})
        print(f"Model: {model_name}, SISDR: {means['sisdr']}, STFT_loss: {means['STFT_loss']}, VisQOL: {means['visqol']}")

        for idx in range(len(samples)):
            full_rows.append({
                "model": model_name,
                "sample_index": idx,
                "sisdr": stats["sisdr"][idx],
                "STFT_loss": stats["STFT_loss"][idx],
                "visqol": stats["visqol"][idx],
            })

    # write CSVs
    with open("results_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_summary)
        writer.writeheader()
        writer.writerows(summary_rows)

    with open("results_per_sample.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_full)
        writer.writeheader()
        writer.writerows(full_rows)

    # Plot metrics per model
    for metric in ["sisdr", "STFT_loss", "visqol"]:
        plt.figure()
        for model_name, stats in all_results.items():
            plt.plot(stats[metric], label=model_name)
        plt.title(f"{metric} across samples")
        plt.xlabel("Sample index")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{metric}_plot.png")
        plt.close()

    

if __name__ == "__main__":
    main()

