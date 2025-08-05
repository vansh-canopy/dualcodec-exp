model_id_to_fname = {
    "12hz_v1": "dualcodec_12hz_16384_4096.safetensors",
    "25hz_v1": "dualcodec_25hz_16384_1024.safetensors",
}
model_id_to_cfgname = {
    "12hz_v1": "dualcodec_12hz_16384_4096_8vq.yaml",
    "25hz_v1": "dualcodec_25hz_16384_1024_12vq.yaml",
}

import warnings
import hydra
from hydra import initialize
from cached_path import cached_path
import os
import safetensors
import safetensors.torch

def get_model(model_id="12hz_v1", pretrained_model_path="hf://amphion/dualcodec", name=None, strict=False):
    with initialize(version_base="1.3", config_path="../../conf/model"):
        cfg = hydra.compose(config_name=model_id_to_cfgname[model_id], overrides=[])
        model = hydra.utils.instantiate(cfg.model)

    if pretrained_model_path is None:        
        warnings.warn(
            "pretrained_model_path is not given, model will be loaded without weights"
        )
    else:
        pretrained_model_path = cached_path(pretrained_model_path)
        print("pretrained_model_path", pretrained_model_path)
        
        if name is None:
            name = model_id_to_fname[model_id]
        
        model_fname = os.path.join(pretrained_model_path, name)
        
        print("Loading model from here", model_fname)
        safetensors.torch.load_model(model, model_fname, strict=strict)
        print("Model loaded")
        
    model.eval()
    return model
