#!/bin/bash

echo "Downloading W2V-BERT 2.0 model..."
huggingface-cli download facebook/w2v-bert-2.0 --local-dir w2v-bert-2.0

echo "Downloading DualCodec checkpoints and W2V-BERT2 statistics..."
huggingface-cli download amphion/dualcodec \
    dualcodec_12hz_16384_4096.safetensors \
    dualcodec_25hz_16384_1024.safetensors \
    discriminator_dualcodec_12hz_16384_4096.safetensors \
    discriminator_dualcodec_25hz_16384_1024.safetensors \
    w2vbert2_mean_var_stats_emilia.pt \
    --local-dir dualcodec_ckpts

echo "All models downloaded successfully!"