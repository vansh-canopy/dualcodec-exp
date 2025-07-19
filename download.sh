#!/bin/bash

echo "Downloading W2V-BERT 2.0 model..."
huggingface-cli download facebook/w2v-bert-2.0 --local-dir w2v-b-2.0

echo "Downloading DualCodec checkpoints and W2V-BERT2 statistics..."
huggingface-cli download amphion/dualcodec \
    dualcodec_12hz_16384_4096.safetensors \
    dualcodec_25hz_16384_1024.safetensors \
    discriminator_dualcodec_12hz_16384_4096.safetensors \
    discriminator_dualcodec_25hz_16384_1024.safetensors \
    w2vbert2_mean_var_stats_emilia.pt \
    --local-dir dualcodec_ckpts

echo "Downloading Whisper causal-base model (config + weights)..."
huggingface-cli download vanshjjw/whisper-stream-lookahead-3 --local-dir whisper-stream-lookahead-3

echo "Downloading Whisper base extractor configs..."
huggingface-cli download openai/whisper-base \
    feature_extractor_config.json preprocessor_config.json \
    --local-dir whisper-base-conf

echo "All models downloaded successfully!"