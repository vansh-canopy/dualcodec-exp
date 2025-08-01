import torch
import torch.nn.functional as F

__all__ = [
    "sisdr",
    "multiscale_stft_loss",
]

def sisdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if reference.shape[0] == 1:
        reference = reference.squeeze(0).squeeze(0)
    
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    
    alpha = torch.dot(estimate, reference) / (torch.dot(reference, reference) + eps)
    aim = alpha * reference
    noise = estimate - aim
    return 10 * torch.log10((aim.pow(2).sum() + eps) / (noise.pow(2).sum() + eps))


def multiscale_stft_loss(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    FFT_sizes: tuple[int, ...] = (2048, 512),
    eps: float = 1e-8,
) -> torch.Tensor:
    loss = torch.zeros(1, dtype=estimate.dtype, device=estimate.device)
    for n_FFT in FFT_sizes:
        hop = n_FFT // 4
        estimate_STFT = torch.stft(estimate, n_fft=n_FFT, hop_length=hop, win_length=n_FFT, return_complex=True)
        reference_STFT = torch.stft(reference, n_fft=n_FFT, hop_length=hop, win_length=n_FFT, return_complex=True)
        
        estimate_magnitude, reference_magnitude = estimate_STFT.abs(), reference_STFT.abs()
        sc = (reference_magnitude - estimate_magnitude).norm(p="fro") / (reference_magnitude.norm(p="fro") + eps)
        log_magnitude = F.l1_loss(torch.log(reference_magnitude + eps), torch.log(estimate_magnitude + eps))
        loss = loss + sc + log_magnitude
    return loss

