 # DualCodec Model Parameter Guide

This document explains how each parameter in the DualCodec model flows through the architecture and what aspect of the model each parameter controls.

## Architecture Overview

The DualCodec model consists of two main pathways:
1. **Semantic Path**: Uses ConvNeXt blocks for semantic representation
2. **Acoustic Path**: Uses DAC (Descriptive Audio Codec) for acoustic details

## Parameter Flow Analysis

### Core Audio Processing Parameters

#### `encoder_dim`
- **Flow**: Passed to DAC → Encoder → Initial channel dimension
- **Controls**: Starting channel width for the acoustic encoder's convolutional layers
- **Impact**: Affects model capacity and computational cost of acoustic feature extraction

#### `encoder_rates` 
- **Flow**: Passed to DAC → Encoder → Downsampling strides
- **Controls**: Temporal downsampling factor at each encoder stage
- **Impact**: Determines time compression ratio and receptive field size
- **Calculation**: Total downsampling = product of all rates, affects `hop_length`

#### `decoder_dim`
- **Flow**: Passed to DAC → Decoder → Initial channel dimension  
- **Controls**: Starting channel width for the acoustic decoder's upsampling layers
- **Impact**: Affects reconstruction quality and computational cost

#### `decoder_rates`
- **Flow**: Passed to DAC → Decoder → Upsampling strides
- **Controls**: Temporal upsampling factor at each decoder stage  
- **Impact**: Must match encoder rates (in reverse) for proper reconstruction

#### `latent_dim`
- **Flow**: Auto-calculated if None as `encoder_dim * (2 ** len(encoder_rates))`
- **Controls**: Dimensionality of the compressed acoustic representation
- **Impact**: Bottleneck size affecting information capacity and compression ratio

### Quantization Parameters

#### `n_codebooks`
- **Flow**: Passed to DAC → ResidualVectorQuantize → Number of quantization stages
- **Controls**: Depth of residual quantization hierarchy  
- **Impact**: Quality vs bitrate tradeoff - more codebooks = higher quality, higher bitrate

#### `codebook_size`
- **Flow**: Passed to DAC → ResidualVectorQuantize → Vocabulary size per codebook
- **Controls**: Number of discrete vectors in each acoustic codebook
- **Impact**: Affects quantization granularity and bitrate

#### `codebook_dim`
- **Flow**: Passed to DAC → ResidualVectorQuantize → Vector dimension
- **Controls**: Dimensionality of each quantized vector
- **Impact**: Information capacity per quantized token

#### `semantic_codebook_size`
- **Flow**: Used in DualCodec → ResidualVectorQuantize for semantic path
- **Controls**: Number of discrete vectors in the semantic codebook
- **Impact**: Semantic representation granularity, typically larger than acoustic

#### `semantic_codebook_dim`
- **Flow**: Used in DualCodec → ResidualVectorQuantize for semantic path  
- **Controls**: Dimensionality of semantic quantized vectors
- **Impact**: Information capacity of semantic tokens

#### `quantizer_dropout`
- **Flow**: Passed to both acoustic and semantic quantizers
- **Controls**: Probability of dropping quantization during training
- **Impact**: Regularization technique to improve robustness

### Semantic Processing Parameters

#### `convnext_dim`
- **Flow**: Used in DualCodec → ConvNeXt encoder/decoder → Hidden dimension
- **Controls**: Channel width of ConvNeXt transformer blocks
- **Impact**: Capacity for semantic understanding and feature extraction

#### `convnext_layers`
- **Flow**: Used in DualCodec → Number of ConvNeXt blocks in encoder/decoder
- **Controls**: Depth of semantic processing pathway
- **Impact**: Model expressiveness vs computational cost

#### `decode_semantic_for_codec`
- **Flow**: Controls whether semantic features are decoded before passing to DAC
- **Controls**: Whether semantic info is preprocessed or directly subtracted
- **Impact**: Architecture choice affecting how semantic and acoustic paths interact

### Causality and Real-time Parameters

#### `is_causal`
- **Flow**: Passed to ConvNeXt blocks → Controls attention/convolution masking
- **Controls**: Whether future frames can be accessed during processing
- **Impact**: Enables real-time streaming vs offline processing

#### `look_ahead`
- **Flow**: Passed to DAC → Decoder → CausalUpsample layers
- **Controls**: Whether decoder can use limited future context
- **Impact**: Quality vs latency tradeoff in streaming scenarios

### Audio Format Parameters

#### `sample_rate`
- **Flow**: Stored and used throughout for preprocessing and validation
- **Controls**: Expected audio sampling frequency
- **Impact**: Determines temporal resolution and processing requirements

#### `semantic_downsample_factor`
- **Flow**: Stored as instance variable, controls semantic temporal resolution
- **Controls**: Additional downsampling applied to semantic features
- **Impact**: Affects temporal alignment between semantic and acoustic features

### Distillation Parameters

#### `distill_projection_out_dim`
- **Flow**: Passed to DAC → Used if distillation is enabled
- **Controls**: Output dimension of distillation projection layer
- **Impact**: Affects knowledge transfer capabilities when training with distillation

## Parameter Interactions

### Temporal Alignment
- `encoder_rates` and `decoder_rates` must be symmetric
- `semantic_downsample_factor` affects temporal matching between pathways
- Total compression ratio affects memory usage and processing speed

### Quality vs Efficiency
- Larger `convnext_dim` and more `convnext_layers` improve semantic understanding
- More `n_codebooks` with larger `codebook_size` improve reconstruction quality  
- `is_causal` and `look_ahead` control latency vs quality tradeoffs

### Architecture Coupling
- `decode_semantic_for_codec` determines how semantic and acoustic paths integrate
- `latent_dim` serves as the interface dimension between encoder and quantizer
- Codebook dimensions affect the information bottleneck in both pathways

## Usage Recommendations

- For real-time applications: Enable `is_causal`, consider `look_ahead` for quality
- For high-quality offline processing: Disable causality, increase model dimensions
- For low-bitrate applications: Reduce codebook sizes and number of codebooks
- For semantic-rich content: Increase `convnext_dim` and `semantic_codebook_size`