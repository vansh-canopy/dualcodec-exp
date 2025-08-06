from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from audiotools import AudioSignal

from dualcodec.infer.dualcodec.get_model import get_model
from dualcodec.infer.dualcodec.inference_with_semantic import pad_to_length
from dualcodec.infer.dualcodec.inference_with_semantic import Inference as DualCodecInference
from dualcodec.dataset.emilia_hf import _to_mp3
from dualcodec.model_codec.loss import MelSpectrogramLoss
import torchaudio
from metrics import sisdr, multiscale_stft_loss, visqol_score


def build_dataset(FROM_ROW: int = 4300):
    raw_ds = load_dataset("vanshjjw/amu-pushed-luna-4500r", split="train")

    ds = raw_ds.map(
        _to_mp3,
        remove_columns=raw_ds.column_names,
        num_proc=4,
    )

    ds = ds.select(range(FROM_ROW, len(ds)))
    return ds


def evaluate_model(model_path: Path, ds, device, n_quantizers: int = 2):
    model = get_model(
        model_id="12hz_v1",
        pretrained_model_path=str(model_path.parent),
        name=model_path.name,
        strict=False,
    ).to(device)
    model.eval()

    inference = DualCodecInference(dualcodec_model=model, device=device)

    spec_loss_fn = MelSpectrogramLoss(
        pow=2,
        mag_weight=1,
        log_weight=1,
        n_mels=[40, 80, 160, 320],
        window_lengths=[256, 512, 1024, 2048],
    ).to(device)

    # Metric accumulators
    mel_losses = []
    sisdr_scores = []
    stft_losses = []
    visqol_scores = []
    for sample in ds:
        wav_np = sample["mp3"]["array"]
        sr = int(sample["mp3"]["sampling_rate"])
        if isinstance(wav_np, list):
            wav_np = np.asarray(wav_np, dtype=np.float32)
        wav = torch.from_numpy(wav_np).unsqueeze(0).to(torch.float32).to(device)

        with torch.no_grad():
            sem_codes, acu_codes = inference.encode(wav, n_quantizers=n_quantizers)
            recon = inference.decode(sem_codes, acu_codes)
            recon = pad_to_length(recon, wav.shape[-1])
        
        mel_loss_val = spec_loss_fn(AudioSignal(wav, sr), AudioSignal(recon, sr)).item()
        mel_losses.append(mel_loss_val)

        # Prepare tensors for metric computation
        reference = wav.squeeze(0).squeeze(0).detach().cpu()
        prediction = recon.squeeze(0).squeeze(0).detach().cpu()

        # Ensure equal length
        min_len = min(reference.shape[-1], prediction.shape[-1])
        reference = reference[..., :min_len]
        prediction = prediction[..., :min_len]

        sisdr_scores.append(sisdr(prediction, reference).item())
        stft_losses.append(multiscale_stft_loss(prediction, reference).item())

        # Resample to 16 kHz for VisQOL speech mode
        reference_16k = torchaudio.transforms.Resample(sr, 16000)(reference)
        prediction_16k = torchaudio.transforms.Resample(sr, 16000)(prediction)
        
        visqol_scores.append(visqol_score(prediction_16k, reference_16k))
        
    return {
        "mel_loss": float(np.mean(mel_losses)),
        "sisdr": float(np.mean(sisdr_scores)),
        "STFT_loss": float(np.mean(stft_losses)),
        "visqol": float(np.mean(visqol_scores)),
    }



def main():
    device = "cuda:3"

    ds = build_dataset()

    model_dir = Path("averaged_models").absolute()
    safetensor_files = sorted([p for p in model_dir.iterdir() if p.suffix == ".safetensors"])

    if not safetensor_files:
        raise RuntimeError(f"No .safetensors files found in {model_dir}")

    print(f"Evaluating {len(safetensor_files)} models on {len(ds)} samples...")

    for file in safetensor_files:
        metrics = evaluate_model(file, ds, device, n_quantizers=2)
        print(
            f"{file.name}: MelSpecLoss = {metrics['mel_loss']:.4f}, "
            f"SISDR = {metrics['sisdr']:.4f}, "
            f"STFT_loss = {metrics['STFT_loss']:.4f}, "
            f"VisQOL = {metrics['visqol']:.4f}"
        )


if __name__ == "__main__":
    main()
