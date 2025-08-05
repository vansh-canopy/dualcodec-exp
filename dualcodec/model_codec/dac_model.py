import math
from typing import List
from typing import Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn
from torch.nn.utils import weight_norm

from .dac_layers import Snake1d
from .dac_layers import WNConv1d
from .dac_quantize import ResidualVectorQuantize
from easydict import EasyDict as edict
import torch.nn.functional as F
from .cnn import ConvNeXtBlock


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)   # type: ignore


def pad_to_length(x, length, pad_value=0):
    current_length = x.shape[-1]

    if length > current_length:
        pad_amount = length - current_length
        x_padded = F.pad(x, (0, pad_amount), value=pad_value)
    else:
        x_padded = x[..., :length]

    return x_padded


class CausalWNConv1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs = dict(kwargs)
        self.conv = weight_norm(nn.Conv1d(*args, **kwargs))
        self.add_module("conv", self.conv) # so params show up in .named_parameters()

    @property
    def kernel_size(self):
        return self.conv.kernel_size
    @property
    def dilation(self):
        return self.conv.dilation

    def forward(self, x):
        k, d, s = self.kernel_size[0], self.dilation[0], self.conv.stride[0]
        left_pad = (k - 1) * d + (1 - s)
        x = F.pad(x, (left_pad, 0))        
        return self.conv(x)                



class CausalUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, kernel_size: int = 4, look_ahead=False):
        super().__init__()
        self.upsample_by = stride
        
        if look_ahead:
            pad = 0
            self.conv = WNConv1d(in_ch, out_ch, kernel_size, padding=pad)
        else:
            self.conv = CausalWNConv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=1)  
         
    def forward(self, x):
            x = x.repeat_interleave(self.upsample_by, dim=-1)  
            x = self.conv(x)                                 
            return x


class ResidualUnit(nn.Module):
    def __init__(self, input_dimension: int = 16, dilation: int = 1, make_causal: bool = False):
        super().__init__()
        
        if make_causal:
            self.block = nn.Sequential(
                Snake1d(input_dimension),
                CausalWNConv1d(input_dimension, input_dimension, dilation=dilation, kernel_size=7),
                Snake1d(input_dimension),
                CausalWNConv1d(input_dimension, input_dimension, kernel_size=1),
            )
        else:  
            pad = (dilation * (7 - 1) ) // 2
            self.block = nn.Sequential(
                Snake1d(input_dimension),
                WNConv1d(input_dimension, input_dimension, dilation=dilation, kernel_size=7, padding=pad),
                Snake1d(input_dimension),
                WNConv1d(input_dimension, input_dimension, kernel_size=1),
            )

    def forward(self, x):
        y = self.block(x) 
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, output_dimension: int = 16, stride: int = 1):
        super().__init__()
        
        input_dimension = output_dimension // 2    

        self.block = nn.Sequential(
            ResidualUnit(input_dimension, dilation=1),
            ResidualUnit(input_dimension, dilation=3),
            ResidualUnit(input_dimension, dilation=9),
            Snake1d(input_dimension),
            WNConv1d(
                input_dimension,
                output_dimension,
                kernel_size=2 * stride,
                stride=stride
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: list = [2, 4, 8, 8],
        latent_dim: Union[int, None] = None,
    ):
        super().__init__()
        
        self.block = [WNConv1d(1, encoder_dim, kernel_size=7)]

        for stride in encoder_rates:
            encoder_dim = encoder_dim * 2
            self.block += [EncoderBlock(encoder_dim, stride=stride)]

        self.block += [
            Snake1d(encoder_dim),
            WNConv1d(encoder_dim, latent_dim, kernel_size=3),
        ]

        self.block = nn.Sequential(*self.block)
        self.encoder_dim = encoder_dim

    def forward(self, x):
        return self.block(x)  # type: ignore


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, is_causal: bool = False, look_ahead = False):
        super().__init__()
        
        self.block = nn.Sequential(
            Snake1d(input_dim),
            CausalUpsample(
                input_dim,
                output_dim,
                stride=stride,
                kernel_size=2*stride,
                look_ahead=look_ahead
            ),
            ResidualUnit(output_dim, dilation=1, make_causal=is_causal),
            ResidualUnit(output_dim, dilation=3, make_causal=is_causal),
            ResidualUnit(output_dim, dilation=9, make_causal=is_causal),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        is_causal: bool = False,
        look_ahead: bool = False
    ):
        super().__init__()

        # Add first conv layer
        layers = [CausalWNConv1d(input_channel, channels, kernel_size=7)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride, is_causal, look_ahead)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            CausalWNConv1d(output_dim, d_out, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DAC(BaseModel):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: Union[int, None] = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        distill_projection_out_dim=1024,
        distill=False,
        use_convnext=True,
        make_convnext_causal=False,
        make_dac_causal=False,
        add_dac_look_ahead=False
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim, # type: ignore
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            is_causal=make_dac_causal,
            look_ahead=add_dac_look_ahead,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.distill = distill
        if self.distill:
            self.distill_projection = WNConv1d(
                latent_dim,
                distill_projection_out_dim,
                kernel_size=1,
            )
            if use_convnext:
                self.convnext = nn.Sequential(
                    *[
                        ConvNeXtBlock(
                            dim=distill_projection_out_dim,
                            intermediate_dim=2048,
                            is_causal=make_convnext_causal,
                        )
                        for _ in range(5)
                    ],  # Unpack the list directly into nn.Sequential
                    WNConv1d(
                        distill_projection_out_dim,
                        1024,
                        kernel_size=1,
                    ),
                )
            else:
                self.convnext = nn.Identity()


    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data


    def encode(
        self,
        audio_data: torch.Tensor,
        sample_rate=24000,
        n_quantizers: Union[int, None] = None,
        subtracted_latent=None,
    ):
        assert not self.training
        audio_data = self.preprocess(audio_data, sample_rate)
        z = self.encoder(audio_data)
        if subtracted_latent is not None:
            assert np.abs(z.shape[-1] - subtracted_latent.shape[-1]) <= 2
            z = z[..., : subtracted_latent.shape[-1]] - subtracted_latent
        z, codes, latents, commitment_loss, codebook_loss, first_layer_quantized = (
            self.quantizer(
                z,
                n_quantizers,
                possibly_no_quantizer=False,
            )
        )
        if subtracted_latent is not None:
            z = z + subtracted_latent
        return z, codes, latents, commitment_loss, codebook_loss, first_layer_quantized

    def decode_from_codes(self, acoustic_codes: torch.Tensor, semantic_latent):
        # acoustic codes should not contain any semantic code
        z = 0.0
        if acoustic_codes is not None:
            z = self.quantizer.from_codes(acoustic_codes)[0]
        z = z + semantic_latent

        z = self.decoder(z)  # audio
        return z

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
        subtracted_latent=None,
        bypass_quantize=False,
        possibly_no_quantizer=False,
    ):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z = self.encoder(audio_data)
        
        if subtracted_latent is not None:
            assert (z.shape[-1] - subtracted_latent.shape[-1]) <= 2
            z = z[..., : subtracted_latent.shape[-1]] - subtracted_latent
      
        if bypass_quantize:
            codes, latents, commitment_loss, codebook_loss, first_layer_quantized = (
                None,
                None,
                0.0,
                0.0,
                None,
            )
            z = 0.0
        else:
            z, codes, latents, commitment_loss, codebook_loss, first_layer_quantized = (
                self.quantizer(
                    z,
                    n_quantizers,
                    possibly_no_quantizer=possibly_no_quantizer,
                )
            )
        
        if subtracted_latent is not None:
            z = z + subtracted_latent

        x = self.decoder(z)

        x = pad_to_length(x, length)

        if self.distill:
            first_layer_quantized = self.distill_projection(first_layer_quantized)
            first_layer_quantized = self.convnext(first_layer_quantized)

        return edict(
            {
                "x": x,
                "z": z,
                "codes": codes,
                "latents": latents,
                "penalty": commitment_loss,
                "vq/codebook_loss": codebook_loss,
                "metrics": {},
                "first_layer_quantized": first_layer_quantized,
            }
        )

import numpy as np
from functools import partial

if __name__ == "__main__":
    model = DAC().to("cpu")

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    length = 88200 * 2
    x = torch.randn(1, 1, length).to(model.device)
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    out = model(x)["audio"]
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    # Create gradient variable
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)  # type: ignore
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")

    x = AudioSignal(torch.randn(1, 1, 44100 * 60), 44100)
    model.decompress(model.compress(x, verbose=True), verbose=True) # type: ignore
