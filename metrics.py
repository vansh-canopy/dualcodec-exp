import torch
import torch.nn.functional as F
import numpy as np
import os
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2

__all__ = [
    "sisdr",
    "multiscale_stft_loss",
    "visqol_score",
]

def sisdr(recon: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if reference.shape[0] == 1:
        reference = reference.squeeze(0).squeeze(0)
    
    reference = reference - reference.mean()
    recon = recon - recon.mean()
    
    alpha = torch.dot(recon, reference) / (torch.dot(reference, reference) + eps)
    aim = alpha * reference
    noise = recon - aim
    return 10 * torch.log10((aim.pow(2).sum() + eps) / (noise.pow(2).sum() + eps))


def multiscale_stft_loss(
    recon: torch.Tensor,
    reference: torch.Tensor,
    FFT_sizes: tuple[int, ...] = (2048, 512),
    eps: float = 1e-8,
) -> torch.Tensor:
    loss = torch.zeros(1, dtype=recon.dtype, device=recon.device)
    for n_FFT in FFT_sizes:
        hop = n_FFT // 4
        recon_STFT = torch.stft(recon, n_fft=n_FFT, hop_length=hop, win_length=n_FFT, return_complex=True)
        reference_STFT = torch.stft(reference, n_fft=n_FFT, hop_length=hop, win_length=n_FFT, return_complex=True)
        
        recon_magnitude, reference_magnitude = recon_STFT.abs(), reference_STFT.abs()
        sc = (reference_magnitude - recon_magnitude).norm(p="fro") / (reference_magnitude.norm(p="fro") + eps)
        log_magnitude = F.l1_loss(torch.log(reference_magnitude + eps), torch.log(recon_magnitude + eps))
        loss = loss + sc + log_magnitude
    return loss


def visqol_score(recon: torch.Tensor, reference: torch.Tensor, mode: str = "speech"):
    conf = visqol_config_pb2.VisqolConfig()

    if mode == "audio":
        conf.audio.sample_rate = 48000
        conf.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        conf.audio.sample_rate = 16000
        conf.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")

    conf.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)

    api = visqol_lib_py.VisqolApi()
    api.Create(conf)
    
     # Convert torch tensors to numpy arrays
    # The arrays should be in float64 format with values in the range of int16
    reference_np = reference.numpy().astype(np.float64)
    recon_np = recon.numpy().astype(np.float64)
    
    # Scale if needed (assuming input is in [-1, 1] range)
    if reference_np.max() <= 1.0 and reference_np.min() >= -1.0:
        reference_np = reference_np * 32767.0
        recon_np = recon_np * 32767.0

    similarity = api.Measure(reference_np, recon_np)
    return similarity.moslqo