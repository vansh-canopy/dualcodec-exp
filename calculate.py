#!/usr/bin/env python3
"""
calculate.py – Receptive-field analysis for FaceTokenizer.

Analyzes conv layers and calculates receptive fields from loaded model checkpoints.

Usage:
  python calculate.py --model_path runs/model.pt --mode encoder --t 0
  python calculate.py --model_path runs/model.pt --mode decoder --t 5
  python calculate.py --model_path runs/model.pt --mode combined --t 0
"""
import argparse
from typing import Any, Dict, List, Tuple

import torch.nn as nn
from dualcodec.model_codec.dac_model import DAC, CausalWNConv1d, CausalUpsample
from dualcodec.model_codec.dac_layers import WNConv1d
from math import floor


# Type alias for layer information
Layer = Dict[str, Any]

# Layer type constants
CONV_LAYER = 'C'
REPEAT_LAYER = 'R'


def extract_conv_info(module: nn.Module):
    if isinstance(module, CausalWNConv1d):
        conv = module.conv
        return conv, 0, 0
    elif isinstance(module, nn.Conv1d):
        padding = getattr(module, 'padding', (0,))[0]
        k = getattr(module, 'kernel_size', 1)[0]
        s = getattr(module, 'kernel_size', 1)[0]
        d = getattr(module, 'dilation', 1)[0]
        lookahead = (k-1)*d + (1-s) - padding
        return module, lookahead, padding
    else:
        return None, 0, 0




def add_conv_layer(layers, name, module, layer_type='C', override_stride=None):
    conv, lookahead, _ = extract_conv_info(module)
    if conv is None:
        return
    k, s, d = conv.kernel_size[0], conv.stride[0], conv.dilation[0]
    if override_stride is not None:
        s = override_stride
    
    # Get actual padding from the module
    if isinstance(module, CausalWNConv1d):
        # For causal convolutions, get the actual padding used
        actual_padding = getattr(conv, 'padding', (0,))[0]
    else:
        # For regular Conv1d, get padding directly
        actual_padding = getattr(module, 'padding', (0,))[0]
    
    layers.append({
        'name': name,
        'type': layer_type,
        'k': k, 's': s, 'd': d,
        'lookahead': lookahead,
        'in_ch': conv.in_channels,
        'out_ch': conv.out_channels,
        'groups': conv.groups,
        'padding': actual_padding,
    })


def add_repeat_layer(layers, name, module):
    # For CausalUpsample, use upsample_by
    if isinstance(module, CausalUpsample):
        repeats = module.upsample_by
    else:
        repeats = getattr(module, 'repeats', 2)
    layers.append({
        'name': name,
        'type': 'R',
        'k': 1, 's': repeats, 'd': 1,
        'lookahead': 0,
        'in_ch': 0, 'out_ch': 0,
        'groups': 1,
        'padding': 0,
    })


def get_model_part(model: DAC, mode: str) -> nn.Module:
    if mode.lower() == "encoder":
        return model.encoder
    elif mode.lower() == "decoder":
        return model.decoder
    else:
        return model


def extract_layers(model: DAC, mode: str):
    layers = []
    processed_paths = set()
    target_model = get_model_part(model, mode)
    prefix = mode.lower() if mode in ["encoder", "decoder"] else ""
    for name, module in target_model.named_modules():
        if not name:
            continue
        full_name = f"{prefix}.{name}" if prefix else name
        # CausalUpsample is a repeat/upsample layer
        if isinstance(module, CausalUpsample):
            add_repeat_layer(layers, full_name, module)
        elif isinstance(module, CausalWNConv1d) and full_name not in processed_paths:
            add_conv_layer(layers, full_name, module)
            processed_paths.add(full_name)
            processed_paths.add(f"{full_name}.conv")
        elif isinstance(module, nn.Conv1d) and full_name not in processed_paths:
            if not full_name.endswith('.conv'):
                add_conv_layer(layers, full_name, module)
                processed_paths.add(full_name)
    return layers


def annotate_receptive_field(layers: List[Layer], t_out: int):
    """Annotate layers with receptive field information working backwards from t_out."""
    print(f"t_out: {t_out}")
    n = len(layers)
    
    # Calculate jumps (product of strides up to each layer)
    jumps = []
    J = 1.0
    for layer in layers:
        jumps.append(J)
        if layer["type"] == CONV_LAYER:
            J *= layer["s"]
        elif layer["type"] == REPEAT_LAYER:
            J /= layer["s"]  # Upsample reduces jump
    
    # Calculate p = (k-1)*d - lookahead for each layer
    for layer in layers:
        k, d, s = layer["k"], layer["d"], layer["s"]
        lookahead = layer.get("lookahead", 0)
        
        if lookahead == "symmetric":
            total_pad = (k - 1) * d + (1 - s)
            layer["p"] = floor(total_pad / 2)
        else:
            lookahead_val = lookahead if isinstance(lookahead, int) else 0
            layer["p"] = (k - 1) * d - lookahead_val

    # Work backwards to compute L and R
    L = R = t_out
    for i in reversed(range(n)):
        layer = layers[i]
        k, s, d, p = layer["k"], layer["s"], layer["d"], layer["p"]
        layer_type = layer["type"]
        
        if layer_type == CONV_LAYER:
            L_new = s * L - p
            R_new = s * R + (k - 1) * d - p
        elif layer_type == REPEAT_LAYER:
            L_new = floor(L / s)
            R_new = floor(R / s)
        else:
            L_new = L
            R_new = R
            
        rf_len = R_new - L_new + 1
        layer.update({
            'L': L_new, 'R': R_new, 'rf_len': rf_len, 'jump': jumps[i]
        })
        L, R = L_new, R_new


def print_analysis_table(layers: List[Layer], title: str, t_val: int):
    """Print formatted analysis table."""
    headers = ["Layer", "Type", "In", "Out", "k", "s", "d", "padding", "lookahead", 
               "jump", "L", "R", "rf_len"]
    
    print(f"\n{title} (t = {t_val})")
    print("=" * 180)
    print(f"{headers[0]:<65} {headers[1]:<4} {headers[2]:>4} {headers[3]:>4} "
          f"{headers[4]:>3} {headers[5]:>3} {headers[6]:>3} {headers[7]:>6} {headers[8]:>9} "
          f"{headers[9]:>10} {headers[10]:>9} {headers[11]:>9} {headers[12]:>6}")
    print("-" * 180)
    
    for layer in layers:
        lookahead_str = str(layer.get('lookahead', 0))
        print(f"{layer['name']:<65} {layer['type']:<4} {layer['in_ch']:>4} {layer['out_ch']:>4} "
              f"{layer['k']:>3} {layer['s']:>3} {layer['d']:>3} {layer['padding']:>6} {lookahead_str:>9} "
              f"{layer['jump']:>10.6f} {layer['L']:>9} {layer['R']:>9} {layer['rf_len']:>6}")


def load_model():
    model = DAC(
        encoder_dim=32,
        encoder_rates=[4, 5, 6, 8, 2],
        decoder_dim=1536,
        decoder_rates=[2, 8, 6, 5, 4],
        n_codebooks=7,
        codebook_size=4096,
        codebook_dim=8,
        quantizer_dropout=1.0,
        sample_rate=24000,
        distill_projection_out_dim=1024,
        distill=False,
        use_convnext=True,
        make_convnext_causal=True,
        make_dac_causal=False,
        add_dac_look_ahead=True
    )
    return model


def analyze_model(mode: str, t_val: int):
    model = load_model()
    
    if mode == "combined":
        enc_layers = extract_layers(model, "encoder")
        dec_layers = extract_layers(model, "decoder")
        combined_layers = enc_layers + dec_layers
        
        annotate_receptive_field(combined_layers, t_val)
        print_analysis_table(combined_layers, "COMBINED (ENCODER + DECODER)", t_val)
        
        # Reference tables
        print("\n" + "=" * 80)
        print("INDIVIDUAL PARTS FOR REFERENCE:")
        print("=" * 80)
        
        annotate_receptive_field(enc_layers, t_val)
        print_analysis_table(enc_layers, "ENCODER (reference)", t_val)
        
        annotate_receptive_field(dec_layers, t_val)
        print_analysis_table(dec_layers, "DECODER (reference)", t_val)
    else:
        layers = extract_layers(model, mode)
        annotate_receptive_field(layers, t_val)
        print_analysis_table(layers, mode.upper(), t_val)
    
    print("\nNOTE: C=conv, R=repeat/upsample. Lookahead values from config.")


def main():
    parser = argparse.ArgumentParser(
        description="Receptive field analysis for FaceTokenizer"
    )
    parser.add_argument("--mode", choices=["encoder", "decoder", "combined"], 
                       default="combined",
                       help="Which part to analyze")
    parser.add_argument("--t", type=int, default=0,
                       help="Output index for receptive field analysis")
    
    args = parser.parse_args()
    analyze_model(args.mode, args.t)


if __name__ == "__main__":
    main()
