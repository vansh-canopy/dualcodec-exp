#!/usr/bin/env python3
"""
calculate.py – Receptive-field tables for Descript-Audio-Codec (DAC).

Shows EVERY conv:   EncBlock2-Unit3-Conv1 / Conv2  …  DecBlock4-Unit2-Conv1 / Conv2
Usage
-----
python calculate.py                    # default indices
python calculate.py --t-enc 23 --t-dec 7
"""
import argparse
from typing import Any, Dict, List

import torch.nn as nn
from dualcodec.model_codec.dac_model import DAC                    
from math import ceil, floor


Layer = Dict[str, Any]


def _add_layer(lst: List[Layer], name: str, mod: nn.Module, ltype: str) -> None:
    lst.append(
        dict(
            name=name,
            type=ltype,                      # 'conv' | 'tconv'
            k=mod.kernel_size[0],
            s=mod.stride[0],
            d=mod.dilation[0],
            p=mod.padding[0],
            in_ch=mod.in_channels,
            out_ch=mod.out_channels,
        )
    )



def collect_encoder_layers(enc: nn.Module, strides: List[int]) -> List[Layer]:
    layers: List[Layer] = []

    # 0. input conv
    _add_layer(layers, "EncInputConv", enc.block[0], "conv")

    idx = 1
    for blk_no, stride in enumerate(strides, 1):
        enc_blk = enc.block[idx]         # EncoderBlock N
        idx += 1

        unit_no = 0
        for sub in enc_blk.block:
            # ResidualUnit
            if hasattr(sub, "block"):    # picks up ResidualUnit
                unit_no += 1
                _add_layer(
                    layers,
                    f"EncBlock{blk_no}-Unit{unit_no}-Conv1",
                    sub.block[1],
                    "conv",
                )
                _add_layer(
                    layers,
                    f"EncBlock{blk_no}-Unit{unit_no}-Conv2",
                    sub.block[3],
                    "conv",
                )
            # down-sampling conv is plain Conv1d
            elif isinstance(sub, nn.Conv1d):
                _add_layer(
                    layers,
                    f"EncBlock{blk_no}-DownsampleConv",
                    sub,
                    "conv",
                )

    _add_layer(layers, "EncFinalConv", enc.block[-1], "conv")
    return layers


def collect_decoder_layers(dec: nn.Module, rates: List[int]) -> List[Layer]:
    layers: List[Layer] = []

    _add_layer(layers, "DecInputConv", dec.model[0], "conv")

    cursor = 1
    for blk_no, _ in enumerate(rates, 1):
        dec_blk = dec.model[cursor]      # DecoderBlock N
        cursor += 1

        _add_layer(
            layers,
            f"DecBlock{blk_no}-UpsampleConv",
            dec_blk.block[1],
            "tconv",
        )

        unit_no = 0
        for sub in dec_blk.block:
            if hasattr(sub, "block"):    # ResidualUnit
                unit_no += 1
                _add_layer(
                    layers,
                    f"DecBlock{blk_no}-Unit{unit_no}-Conv1",
                    sub.block[1],
                    "conv",
                )
                _add_layer(
                    layers,
                    f"DecBlock{blk_no}-Unit{unit_no}-Conv2",
                    sub.block[3],
                    "conv",
                )

    _add_layer(layers, "DecFinalConv", dec.model[-2], "conv")
    return layers


def annotate_for_index(layers: List[Layer], t_out: int) -> None:
    """
    Annotate each layer with receptive field info, working backwards from t_out.
    Jumps are computed as the product of strides up to that layer.
    L and R are computed recursively from the output index.
    """
    print(f"t_out: {t_out}")
    n = len(layers)
    # Compute jumps for each layer (J_l = product of strides up to l)
    jumps = []
    J = 1.0
    for layer in layers:
        jumps.append(J)
        if layer["type"] == "conv":
            J *= layer["s"]
        else:
            J /= layer["s"]

    # Work backwards to compute L and R for each layer
    L = R = t_out
    prev_len = 1
    for i in reversed(range(n)):
        layer = layers[i]
        k, s, d, p = layer["k"], layer["s"], layer["d"], layer["p"]
        J = jumps[i]
        if layer["type"] == "conv":
            L_new = L - p * J
            R_new = R + ((k - 1) * d - p) * J
        else:  # transposed conv
            # Use the same formula as in the original code for tconv
            L_new = ceil(L/J + (p - (k - 1) * d) / s) * J
            R_new = floor(R/J + p / s) * J
        rf_len = R_new - L_new + 1
        layer.update(
            dict(jump=J, L=L_new, R=R_new, rf_len=rf_len, delta_len=rf_len - prev_len)
        )
        prev_len = rf_len
        L, R = L_new, R_new

# ─────────────────────────────── printing ─────────────────────────────── #
def print_table(layers: List[Layer], title: str, t_val: int) -> None:
    hdr = ("Layer", "In", "Out", "k", "s", "d", "pad",
           "jump", "L", "R", "len", "Δlen")
    print(f"\n{title}  (t = {t_val})\n" + "=" * 156)
    print(f"{hdr[0]:<50} {hdr[1]:>4} {hdr[2]:>4} "
          f"{hdr[3]:>3} {hdr[4]:>3} {hdr[5]:>3} {hdr[6]:>4} "
          f"{hdr[7]:>10} {hdr[8]:>9} {hdr[9]:>9} {hdr[10]:>6} {hdr[11]:>6}")
    print("-" * 156)
    for l in layers:
        print(f"{l['name']:<50} {l['in_ch']:>4} {l['out_ch']:>4} "
              f"{l['k']:>3} {l['s']:>3} {l['d']:>3} {l['p']:>4} "
              f"{l['jump']:>10.6f} {l['L']:>9} {l['R']:>9} "
              f"{l['rf_len']:>6} {l['delta_len']:>6}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Receptive-field tables for DAC")
    ap.add_argument("--t-enc", type=int, default=0,
                    help="encoder output index (latent frame)")
    ap.add_argument("--t-dec", type=int, default=0,
                    help="decoder output index (audio sample)")
    args = ap.parse_args()

    dac = DAC(decoder_rates=[2,8,6,5,4]).eval()

    enc_layers = collect_encoder_layers(dac.encoder, dac.encoder_rates)
    annotate_for_index(enc_layers, args.t_enc)
    print_table(enc_layers, "ENCODER", args.t_enc)

    dec_layers = collect_decoder_layers(dac.decoder, dac.decoder_rates)
    annotate_for_index(dec_layers, args.t_dec)
    print_table(dec_layers, "DECODER", args.t_dec)

    print("These are in causal units without the shift. That is, we think that the 1th token in the last layer actually has position jumo.")


if __name__ == "__main__":
    main()