from .cnn import ConvNeXtBlock
from .dac_model import DAC
import torch.nn as nn
from typing import List
from typing import Union

import torch
from torch import nn

from .dac_layers import WNConv1d
from .dac_quantize import ResidualVectorQuantize
from easydict import EasyDict as edict
import random
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin


class DualCodec(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: Union[int, None] = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        semantic_codebook_size: int = 16384,
        codebook_dim: Union[int, list] = 8,
        semantic_codebook_dim=8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        distill_projection_out_dim=1024,
        convnext_dim=768,
        convnext_layers=4,
        decode_semantic_for_codec=True,
        make_convnext_causal=False,
        make_dac_causal=True,
        add_dac_look_ahead=False,
        semantic_downsample_factor=2,
        semantic_repr_dim=1024,
    ):
        self.semantic_downsample_factor = semantic_downsample_factor
        super().__init__()

        self.dac = DAC(
            encoder_dim,
            encoder_rates,
            latent_dim,
            decoder_dim,
            decoder_rates,
            n_codebooks,
            codebook_size,
            codebook_dim,
            quantizer_dropout,
            sample_rate,
            distill_projection_out_dim,
            distill=False,
            make_convnext_causal=True,  # convnext inside DAC should always be causal?
            make_dac_causal=make_dac_causal,
            add_dac_look_ahead=add_dac_look_ahead,
        )
        self.decode_semantic_for_codec = decode_semantic_for_codec
        self.encoder_rates = encoder_rates
        
    
        if semantic_repr_dim == 1024:
            self.semantic_mapper = nn.Identity()
        else:
            self.semantic_mapper = nn.Linear(semantic_repr_dim, 1024, bias=False)

        self.convnext_encoder = nn.Sequential(
            WNConv1d(1024, convnext_dim, kernel_size=1),
            *[
                ConvNeXtBlock(dim=convnext_dim, intermediate_dim=2048, is_causal=make_convnext_causal)
                for _ in range(convnext_layers)
            ],  # Unpack the list directly into nn.Sequential
        )
        
        self.semantic_vq = ResidualVectorQuantize(
            convnext_dim,
            n_codebooks=1,
            codebook_size=semantic_codebook_size,
            codebook_dim=semantic_codebook_dim,
        )
        
        self.convnext_decoder = nn.Sequential(
            *[
                ConvNeXtBlock(
                    dim=convnext_dim,
                    intermediate_dim=2048,
                    is_causal=make_convnext_causal,
                )
                for _ in range(convnext_layers)
            ],  # Unpack the list directly into nn.Sequential
            WNConv1d(convnext_dim, 1024, kernel_size=1),  # keep decoder output at 1024 for DAC
        )
        
        if not self.decode_semantic_for_codec:
            assert convnext_dim == 1024

        # Inverse mapper for distillation: map 1024-ch back to original semantic_repr_dim
        if semantic_repr_dim == 1024:
            self.semantic_inverse_mapper = nn.Identity()
        else:
            self.semantic_inverse_mapper = nn.Linear(1024, semantic_repr_dim, bias=False)

    def semantic_quantize(self, semantic_repr):
        # semantic_repr: (B, C, T)  →  (B, T, C) for Linear → back to (B, C, T)
        if semantic_repr is not None and not isinstance(self.semantic_mapper, nn.Identity):
            semantic_repr = semantic_repr.transpose(1, 2)
            semantic_repr = self.semantic_mapper(semantic_repr)
            semantic_repr = semantic_repr.transpose(1, 2)
        
        semantic = self.convnext_encoder(semantic_repr)
        semantic, codes, _, _, _, _ = self.semantic_vq(semantic)
        codes = rearrange(codes, "b 1 t -> b t")
        return codes

    def encode(
        self, audio_data, num_quantizers=None, sample_rate=24000, semantic_repr=None
    ):
        assert not self.training
        if semantic_repr is not None and not isinstance(self.semantic_mapper, nn.Identity):
            semantic_repr = semantic_repr.transpose(1, 2)
            semantic_repr = self.semantic_mapper(semantic_repr)
            semantic_repr = semantic_repr.transpose(1, 2)
        
        semantic = self.convnext_encoder(semantic_repr)
        semantic, codes, _, _, _, _ = self.semantic_vq(semantic)
        
        if self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)
        semantic_codes = codes

        if num_quantizers == 1:
            return semantic_codes, None

        if num_quantizers is not None:
            num_quantizers -= 1

        acoustic_codes = self.dac.encode(
            audio_data,
            sample_rate=sample_rate,
            n_quantizers=num_quantizers,
            subtracted_latent=semantic,
        )[1]
        return semantic_codes, acoustic_codes  # [B, n_q, T]

    @torch.no_grad()
    def decode_from_codes(self, semantic_codes, acoustic_codes):
        """both [B, n_q, T]"""
        semantic = self.semantic_vq.from_codes(semantic_codes)[0]
        if self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)

        audio = self.dac.decode_from_codes(acoustic_codes, semantic)
        return audio

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = 24000,
        n_quantizers: Union[int, None] = None,
        semantic_repr=None,
        bypass_quantize_rate=0.125,
        possibly_no_quantizer=False,
    ):
        if semantic_repr is not None and not isinstance(self.semantic_mapper, nn.Identity):
            semantic_repr = semantic_repr.transpose(1, 2)
            semantic_repr = self.semantic_mapper(semantic_repr)
            semantic_repr = semantic_repr.transpose(1, 2)
        
        semantic = self.convnext_encoder(semantic_repr)
        (
            semantic,
            codes,
            latents,
            commitment_loss,
            codebook_loss,
            _,
        ) = self.semantic_vq(semantic)
        if self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)

        bypass_quantize = random.random() < bypass_quantize_rate
        
        if not self.training:
            bypass_quantize = False

        if n_quantizers == 1:
            bypass_quantize = True
        
        if n_quantizers is not None:
            n_quantizers = n_quantizers - 1
        
        acoustic_edict = self.dac(
            audio_data,
            sample_rate,
            n_quantizers,
            subtracted_latent=semantic,
            bypass_quantize=bypass_quantize,
            possibly_no_quantizer=possibly_no_quantizer,
        )
        
        if not self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)

        # Map semantic back to original dimension for distillation logging
        semantic_out = semantic
        if not isinstance(self.semantic_inverse_mapper, nn.Identity):
            semantic_out = semantic_out.transpose(1,2)  # (B,T,1024)
            semantic_out = self.semantic_inverse_mapper(semantic_out)  # (B,T,orig_dim)
            semantic_out = semantic_out.transpose(1,2)

        semantic_edict = edict(
            {
                "x": semantic_out,
                "codes": codes,
                "latents": latents,
                "penalty": commitment_loss,
                "vq/codebook_loss": codebook_loss,
                "metrics": {},
                "bypassed_quantize": bypass_quantize,
            }
        )
        return acoustic_edict, semantic_edict
