import torch
import torchaudio
import dualcodec
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import csv, pathlib
import matplotlib.pyplot as plt

DEVICE: str = "cuda"

from metrics import sisdr, multiscale_stft_loss, visqol_score


def load_audio(path: str, required_sr: int = 24000) -> torch.Tensor:
    sr, data = wavfile.read(path)
    if sr != required_sr:
        gcd = np.gcd(sr, required_sr)
        data = resample_poly(data, required_sr // gcd, sr // gcd)
    return torch.from_numpy(data)

    
    
def load_models():
    base_id = "12hz_v1"
    base_model = dualcodec.get_model(base_id)
    base_inference = dualcodec.Inference(dualcodec_model=base_model)
    
    MODELS: list[tuple[str, dualcodec.Inference]] = [("jianqi_dualcodec", base_inference)]
    
    my_model_path = "https://huggingface.co/vanshjjw/vansh-dualcodec-step-1.030"
    my_model = dualcodec.get_model(base_id, my_model_path, is_checkpoint=True)
    my_inference = dualcodec.Inference(dualcodec_model=my_model)
    MODELS.append(("vansh_dualcodec", my_inference))
    
    DIRECTORY_TO_LOAD_FROM = pathlib.Path("/home/vansh/dualcodec-vansh/averaged_models")

    for path in sorted(DIRECTORY_TO_LOAD_FROM.glob("*.safetensors")):
        filename = path.name  # e.g. averaged_model_step_0795000_decay_0.9.safetensors
        model = dualcodec.get_model(base_id, str(DIRECTORY_TO_LOAD_FROM), name=filename)
        inference_model = dualcodec.Inference(dualcodec_model=model)
        MODELS.append((filename, inference_model))

    return MODELS


def evaluate(model: torch.nn.Module, samples: list[torch.Tensor], device: str = DEVICE) -> dict[str, list[float]]:
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
    
    
SAMPLES_DIR = "/home/vansh/dualcodec-vansh/audio_samples"
OUTPUT_DIR = "/home/vansh/dualcodec-vansh/eval_results"


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

    # ensure output directory exists
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    MODELS = load_models()

    all_results: dict[str, dict[str, list[float]]] = {}
    summary_rows: list[dict[str, float | str]] = []

    for model_name, model in MODELS:
        stats = evaluate(model, samples, device=DEVICE)
        all_results[model_name] = stats

        means = {metric: float(np.mean(values)) for metric, values in stats.items()}
        summary_rows.append({"model": model_name, **means})
        print(f"Model: {model_name}, SISDR: {means['sisdr']:.6f}, STFT_loss: {means['STFT_loss']:.6f}, VisQOL: {means['visqol']:.6f}")

    fieldnames_summary = ["model", "sisdr", "STFT_loss", "visqol"]
    with open(pathlib.Path(OUTPUT_DIR) / "results_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_summary)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Plot metrics per mode
    for metric in ["sisdr", "STFT_loss", "visqol"]:
        plt.figure()
        for model_name, stats in all_results.items():
            plt.plot(stats[metric], label=model_name)
        plt.title(f"{metric} across samples")
        plt.xlabel("Sample index")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(pathlib.Path(OUTPUT_DIR) / f"{metric}.png")
        plt.close()

    

if __name__ == "__main__":
    main()

