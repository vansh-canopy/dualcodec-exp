# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

model_id_to_fname = {
    "12hz_v1": "dualcodec_12hz_16384_4096.safetensors",
    "25hz_v1": "dualcodec_25hz_16384_1024.safetensors",
}
model_id_to_cfgname = {
    "12hz_v1": "dualcodec_12hz_16384_4096_8vq.yaml",
    "25hz_v1": "dualcodec_25hz_16384_1024_12vq.yaml",
}

from cached_path import cached_path


def get_model(model_id="12hz_v1", pretrained_model_path="hf://amphion/dualcodec", name=None, strict=False):
    import os

    # import importlib.resources as pkg_resources
    # conf_dir = pkg_resources.files("dualcodec") / "conf/model"
    pretrained_model_path = cached_path(pretrained_model_path)

    import hydra
    from hydra import initialize

    with initialize(version_base="1.3", config_path="../../conf/model"):
        cfg = hydra.compose(config_name=model_id_to_cfgname[model_id], overrides=[])
        model = hydra.utils.instantiate(cfg.model)

    if pretrained_model_path is None:
        import warnings

        warnings.warn(
            "pretrained_model_path is not given, model will be loaded without weights"
        )
    else:
        if not name:
            name = model_id_to_fname[model_id]
        
        model_fname = os.path.join(pretrained_model_path, name)
        print("Loading model from", model_fname)
        import safetensors.torch
        from safetensors.torch import load_file
        import torch

        # Load tensors from the safetensors file
        pretrained_tensors = load_file(model_fname)

        # Current model parameters
        model_state = model.state_dict()
        filtered_tensors = {}
        skipped_tensors = []
        for k, v in pretrained_tensors.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_tensors[k] = v
            else:
                skipped_tensors.append(k)

        # Load only matching tensors
        missing_keys, unexpected_keys = model.load_state_dict(filtered_tensors, strict=False)
        print(
            f"Loaded {len(filtered_tensors)} tensors, skipped {len(skipped_tensors)} incompatible tensors."
        )
        if missing_keys:
            print(f"Missing keys after load: {len(missing_keys)} (first 10 shown): {missing_keys[:10]}")
        if unexpected_keys:
            print(f"Unexpected keys after load: {len(unexpected_keys)} (first 10 shown): {unexpected_keys[:10]}")
    model.eval()
    return model
