################################################################################
#
# Copyright (c) 2024 Amphion. All Rights Reserved
#
################################################################################

import json
import os
import shutil
import torch
import time
from pathlib import Path
import torch
from tqdm import tqdm
import torch.nn as nn
from dualcodec.utils.base_trainer import BaseTrainer
import safetensors
import numpy as np
from .discriminator import Discriminator
import torch.nn.functional as F
from einops import rearrange
from easydict import EasyDict as edict
import dualcodec

USE_HINGE_LOSS = False
from audiotools import AudioSignal


class Trainer(BaseTrainer):
    """Trainer"""

    def _log(self, msg: str, level: str = "info"):
        """Utility: always prints to stdout and logs via self.logger if present."""
        print(msg)
        if hasattr(self, "logger") and self.logger is not None:
            log_fn = getattr(self.logger, level, None)
            if callable(log_fn):
                log_fn(msg)

    def __init__(self, args=None, cfg=None, **kwargs):
        """
            Initializes the model with the given arguments and configuration.

        Args:
            args (argparse.Namespace, optional): Arguments to be passed on to the model. Defaults to None.
            cfg (dict, optional): Configuration dictionary containing parameters for the model. Defaults to None.
        """
        # Early initialization of weight verification attributes so they exist during BaseTrainer.__init__
        self.weight_check_frequency = 10  # default value; may be overridden later
        self.max_weight_diff_threshold = 1e-8
        self.verbose_weight_check = False
        # store reference snapshots
        self.ref_encoder_state = None
        self.ref_quantizer_state = None

        super().__init__(args, cfg)
        torch.backends.cudnn.benchmark = True
        # Flag to indicate this trainer handles its own backward passes
        self.codec_trainer_handles_backward = True
        
        # Store reference to original model for weight verification
        self.original_model = None
        self.max_weight_diff_threshold = getattr(self.cfg, 'max_weight_diff_threshold', 1e-5)  # Maximum allowed weight difference
        self.verbose_weight_check = getattr(self.cfg, 'verbose_weight_check', False)  # Log detailed weight differences

        from .loss import GANLoss, MelSpectrogramLoss, MultibandMelSpectrogramLoss

        self.gan_loss = GANLoss(self.cfg.discriminator_model)
        self.spec_loss = MelSpectrogramLoss(
            pow=2,
            mag_weight=1,
            log_weight=1,
            n_mels=[40, 80, 160, 320],
            window_lengths=[256, 512, 1024, 2048],
        )
        self.semantic_spec_loss = MultibandMelSpectrogramLoss(
            # bands=[(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)],
            # band_weights=[16,8,4,2,1],
            bands=[(0.0, 0.1)],
            band_weights=[1.0],
            loss_fn=nn.MSELoss(),
            pow=2,
            mag_weight=1,
            log_weight=1,
            n_mels=[80, 160, 320],
            window_lengths=[512, 1024, 2048],
        )

        if hasattr(self.cfg, "semantic_model"):
            for key in self.cfg.semantic_model:
                if isinstance(
                    self.cfg.semantic_model[key], torch.nn.Module
                ) or isinstance(self.cfg.semantic_model[key], torch.Tensor):
                    self.cfg.semantic_model[key] = self.cfg.semantic_model[key].to(
                        self.accelerator.device
                    )
        self.distill = False

        self.model_module = self.model
        if hasattr(self.model, "module"):
            self.model_module = self.model.module
            
        modules_to_freeze = [
            self.model_module.dac.encoder,
            self.model_module.dac.quantizer,
            self.model_module.semantic_vq, 
            self.model_module.convnext_encoder,
        ]
        for m in modules_to_freeze:
            m.eval()     
            for p in m.parameters():
                p.requires_grad = False
        

                # Snapshot current encoder & quantizer weights as reference
        self._load_reference_weights()
        # Verify weights after initialization
        self._verify_encoder_weights("After initialization")

    @torch.no_grad()
    def _extract_semantic_code(self, input_features, attention_mask):
        vq_emb = self.cfg.semantic_model["model"](
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[self.cfg.semantic_model["output_idx"]]  # (B, T, C)

        if (
            hasattr(self.cfg, "skip_semantic_normalize")
            and self.cfg.skip_semantic_normalize
        ):
            pass
        else:
            feat = (feat - self.cfg.semantic_model["mean"]) / self.cfg.semantic_model[
                "std"
            ]
        return feat

    def _build_model(self):
        """
        Returns: None
        """
        return edict(
            {
                "generator": self.cfg.model,
                "discriminator": self.cfg.discriminator_model,
            }
        )

    def _build_optimizer(self):
        r"""Build optimizer for model."""
        return edict(
            {
                "optimizer_g": self.cfg.train.optimizer(
                    params=self.model.generator.parameters()
                ),
                "optimizer_d": self.cfg.train.optimizer(
                    params=self.model.discriminator.parameters()
                ),
            }
        )

    def _accelerator_prepare(self):
        """
        Returns: None
        """
        (
            self.model,
            self.discriminator,
            self.optimizer,
            self.optimizer_d,
        ) = self.accelerator.prepare(
            self.model.generator,
            self.model.discriminator,
            self.optimizer.optimizer_g,
            self.optimizer.optimizer_d,
        )

    def _build_scheduler(self):
        """
        Returns: None
        """
        return None

    @torch.no_grad()
    def _load_reference_weights(self):
        """Load encoder & quantizer weights from a safetensors checkpoint located in dualcodec_ckpts."""
        from safetensors.torch import load_file
        # determine checkpoint path
        ckpt_dir = getattr(self.cfg, 'reference_ckpt_dir', '/home/vansh/dualcodec-exp/dualcodec_ckpts')
        ckpt_name = getattr(self.cfg, 'reference_ckpt_name', 'dualcodec_12hz_16384_4096.safetensors')
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.isfile(ckpt_path):
            self._log(f"Reference checkpoint {ckpt_path} not found; weight verification disabled", level='warning')
            self.ref_encoder_state = None
            self.ref_quantizer_state = None
            return
        tensors = load_file(ckpt_path)
        
        self._log(f"Loaded reference weights from {ckpt_path}")

        # Extract only encoder parameters from checkpoint
        enc_state = {}
        prefix = 'dac.encoder.'
        for k, v in tensors.items():
            if k.startswith(prefix):
                stripped_key = k[len(prefix):]  # remove leading 'dac.encoder.' so keys match state_dict()
                enc_state[stripped_key] = v.cpu()
        if len(enc_state) == 0:
            self._log("Failed to extract reference encoder weights from checkpoint", level='warning')
        self.ref_encoder_state = enc_state
        # quantizer comparison removed



    @torch.no_grad()
    def _verify_encoder_weights(self, checkpoint_name=""):
        if self.ref_encoder_state is None or self.ref_quantizer_state is None:
            return
        
        self._log(f"=== Weight Verification {checkpoint_name} ===")
        
        current_model = self.model_module if not hasattr(self.model, "module") else self.model.module
        
        modules_to_check = [
            ('dac.encoder', current_model.dac.encoder.state_dict(), self.ref_encoder_state),
        ]
        
        all_weights_match = True
        
        for module_name, curr_state, ref_state in modules_to_check:
            module_matches = self._compare_module_weights(
                curr_state,
                ref_state,
                module_name
            )
            if not module_matches:
                all_weights_match = False
        
        if all_weights_match:
            self._log(f"✓ Encoder & quantizer match reference {checkpoint_name}")
        else:
            self._log(f"✗ Weight mismatch detected versus reference {checkpoint_name}")
        
        self._log("=" * 50)
        
    def _compare_module_weights(self, current_state_dict, original_state_dict, module_name):
        """
        Compare weights between current and original module.
        
        Returns:
            bool: True if all weights are within threshold, False otherwise
        """
        all_close = True
        max_diff_overall = 0.0
        max_diff_param = ""
        
        for key in current_state_dict.keys():
            if key not in original_state_dict:
                all_close = False
                continue
            
            current_param = current_state_dict[key].detach().cpu()
            original_param = original_state_dict[key].detach().cpu()
            
            # Check shape match
            if current_param.shape != original_param.shape:
                self._log(f"[{module_name}] Shape mismatch for {key}: {current_param.shape} vs {original_param.shape}")
                all_close = False
                continue
            
            # Calculate difference on CPU to avoid device mismatch
            diff = torch.abs(current_param - original_param)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            if max_diff > max_diff_overall:
                max_diff_overall = max_diff
                max_diff_param = f"{module_name}.{key}"
            
            # Log if difference exceeds threshold
            if max_diff > self.max_weight_diff_threshold:
                self._log(
                    f"[{module_name}] {key} - Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e} - EXCEEDS THRESHOLD"
                )
                all_close = False
            elif self.verbose_weight_check:  # Only log details if verbose
                self._log(
                    f"[{module_name}] {key} - Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}"
                )
        
        if max_diff_overall > 0:
            self._log(f"[{module_name}] Max difference: {max_diff_overall:.2e} in {max_diff_param}")
        
        return all_close
    
    def _verify_frozen_gradients(self):
        """Verify that frozen modules have no gradients."""

        current_model = self.model_module
        if hasattr(self.model, "module"):
            current_model = self.model.module
        
        modules_to_check = [
            ('dac.encoder', current_model.dac.encoder),
            ('dac.quantizer', current_model.dac.quantizer),
            ('semantic_vq', current_model.semantic_vq),
            ('convnext_encoder', current_model.convnext_encoder),
        ]
        
        for module_name, module in modules_to_check:
            for param_name, param in module.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    if grad_norm > 1e-8:  # Small threshold for numerical errors
                        self._log(
                            f"[{module_name}] {param_name} has non-zero gradient: {grad_norm:.2e}"
                        )

    def _train_step(self, batch):
        """
        Args:
        - batch: dict containing the batch data
        -- batch["speech"]: torch.Tensor of shape (B, T)
        -- batch["speech_lens"]: torch.Tensor of shape (B,), contains the length of the unpadded speech
        -- batch["input_features"]: torch.Tensor of shape (B, T, C), extracted by w2v-bert feat extractor
        -- batch["attention_mask"]: torch.Tensor of shape (B, T), attention mask for the input_features, extracted by w2v-bert feat extractor
        """
        optim_g, optim_d = self.optimizer, self.optimizer_d

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.accelerator.device)

        x_wav, audio_lengths = batch["speech"], batch["speech_lens"]

        x_wav = x_wav.float()[:, None, :]

        if self.cfg.semantic_vq:
            input_features = batch["input_features"]
            attention_mask = batch["attention_mask"]
            feat = self._extract_semantic_code(
                input_features, attention_mask
            ).transpose(1, 2)
            feat = torch.nn.functional.avg_pool1d(
                feat,
                self.model_module.semantic_downsample_factor,
                self.model_module.semantic_downsample_factor,
            )
            out_dict, semantic_edict = self.model(
                x_wav,
                semantic_repr=feat,
                bypass_quantize_rate=0.125,
                possibly_no_quantizer=False,  # internal dropout
            )
        else:
            out_dict = self.model(
                x_wav,
            )
            semantic_edict = None

        generator_out = out_dict.x
        commitment_loss = out_dict.penalty
        metrics = out_dict.metrics
        if hasattr(out_dict, 'vq/codebook_loss'):
            codebook_loss = out_dict['vq/codebook_loss']
        else:
            codebook_loss = 0.0
        if hasattr(out_dict, 'first_layer_quantized'):
            first_layer_quantized = out_dict['first_layer_quantized']
        else:
            first_layer_quantized = None

        # --------- Discriminator training ------------
        if USE_HINGE_LOSS:
            disc_loss = self.gan_loss.discriminator_hinge_loss(generator_out, x_wav)
        else:
            disc_loss = self.gan_loss.discriminator_loss(generator_out, x_wav)
        self.optimizer_d.zero_grad()
        self.accelerator.backward(disc_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.optimizer_d.step()
        self.optimizer_d.zero_grad()

        if USE_HINGE_LOSS:
            adv_g_loss, feat_loss = self.gan_loss.generator_hinge_loss(
                generator_out, x_wav
            )
        else:
            adv_g_loss, feat_loss = self.gan_loss.generator_loss(generator_out, x_wav)
        spec_loss = self.spec_loss(
            AudioSignal(x_wav, 24000), AudioSignal(generator_out, 24000)
        )
        # spec_loss = reconstruction_loss(x_wav, generator_out, args)
        total_loss = (
            0.25 * commitment_loss
            + 1.5 * adv_g_loss
            + 2.5 * feat_loss
            + 15.0 * spec_loss
            + 1.0 * codebook_loss
        )
        # ---------- Generator training ----------------
        if semantic_edict:
            distill_loss = F.mse_loss(feat, semantic_edict["x"])
            total_loss += (
                distill_loss * self.cfg.lambda_distill_loss
                + self.cfg.lambda_semantic_commitment_loss * semantic_edict["penalty"]
                + self.cfg.lambda_semantic_codebook_loss
                + semantic_edict["vq/codebook_loss"]
            )
            metrics.update(
                {
                    "semantic/semantic_commitment_loss": semantic_edict["penalty"],
                    "semantic/semantic_codebook_loss": semantic_edict[
                        "vq/codebook_loss"
                    ],
                    "semantic/semantic_distill_loss": distill_loss,
                }
            )
            if semantic_edict["bypassed_quantize"]:
                metrics.update(
                    {
                        "semantic/spec_loss": spec_loss,
                    }
                )
            if self.cfg.add_semantic_spec_loss and semantic_edict["bypassed_quantize"]:
                semantic_spec_loss = 15.0 * self.semantic_spec_loss(
                    AudioSignal(x_wav, 24000), AudioSignal(generator_out, 24000)
                )
                total_loss += semantic_spec_loss
                metrics.update(
                    {
                        "semantic/semantic_spec_loss": semantic_spec_loss,
                    }
                )

        if self.distill:
            input_features = batch["input_features"]
            attention_mask = batch["attention_mask"]
            feat = self._extract_semantic_code(
                input_features, attention_mask
            ).transpose(1, 2)
            feat = torch.nn.functional.avg_pool1d(
                feat, self.semantic_downsample_factor, self.semantic_downsample_factor
            )
            distill_loss = F.mse_loss(
                feat, first_layer_quantized[..., : feat.shape[-1]]
            )
            total_loss += distill_loss * self.cfg.lambda_distill_loss
        else:
            distill_loss = 0.0
        
        
        
        self.optimizer.zero_grad()        
        self.accelerator.backward(total_loss)
        
        # Verify that frozen module gradients are None or zero
        if self.step % self.weight_check_frequency == 0:
            self._verify_frozen_gradients()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Periodically verify weights haven't changed
        if self.step % self.weight_check_frequency == 0:
            self._verify_encoder_weights(f"Step {self.step}")

        # print(commitment_loss, spec_loss)
        # print(x_wav.shape[0])
        # breakpoint()
        # print(out_dict.codes)
        # if self.step >= 100:
        #     breakpoint()
        metrics.update(
            {
                "commitment_loss": commitment_loss,
                "spec_loss": spec_loss,
                "feat_loss": feat_loss,
                "adv_g_loss": adv_g_loss,
                "total_loss": total_loss,
                "Train/Batch Size": x_wav.shape[0],
                "disc_loss": disc_loss.item(),
                "distill_loss": distill_loss,
            }
        )
        # print(metrics)

        return total_loss, metrics

    def _load_model(
        self,
        checkpoint_dir: str = None,
        checkpoint_path: str = None,
        resume_type: str = "",
    ):
        r"""Load model from checkpoint. If checkpoint_path is None, it will
        load the latest checkpoint in checkpoint_dir. If checkpoint_path is not
        None, it will load the checkpoint specified by checkpoint_path. **Only use this
        method after** ``accelerator.prepare()``.
        """
        if checkpoint_path is None:
            try:
                all_ckpts = os.listdir(checkpoint_dir)
                all_ckpts = filter(lambda x: x.startswith("epoch"), all_ckpts)
                ls = list(all_ckpts)
                ls = [os.path.join(checkpoint_dir, i) for i in ls]
                ls.sort(
                    key=lambda x: int(x.split("_")[-2].split("-")[-1]), reverse=True
                )
                checkpoint_path = ls[0]
                self._log("Resume from {}".format(checkpoint_path))
            except Exception as e:
                print(
                    "Failed to load checkpoint from {}, starting FROM SCRATCH...".format(
                        checkpoint_dir
                    )
                )
                return None

        if resume_type in ["resume", ""]:
            # Load all the things, including model weights, optimizer, scheduler, and random states.
            try:
                self.accelerator.load_state(input_dir=checkpoint_path)
            except Exception as e:
                print(e)
            # set epoch and step
            from pathlib import Path

            self.epoch = int(Path(checkpoint_path).name.split("_")[0].split("-")[-1])
            if hasattr(self.args, "reset_steps") and self.args.reset_steps:
                self.step = 0
            else:
                self.step = (
                    int(Path(checkpoint_path).name.split("_")[1].split("-")[-1]) + 1
                )

            self._verify_encoder_weights(f"After resuming from {checkpoint_path}")

        elif resume_type == "finetune":
            # Load only the model weights
            import safetensors.torch

            from safetensors.torch import load_file

            # ---- Safe load for generator ----
            gen_ckpt_path = os.path.join(checkpoint_path, self.args.model_1_name)
            gen_tensors = load_file(gen_ckpt_path)
            gen_state = self.accelerator.unwrap_model(self.model).state_dict()
            gen_filtered = {k: v for k, v in gen_tensors.items() if k in gen_state and gen_state[k].shape == v.shape}
            self.accelerator.unwrap_model(self.model).load_state_dict(gen_filtered, strict=False)
            self._log(
                f"Loaded {len(gen_filtered)} compatible tensors into generator from {gen_ckpt_path}. Skipped {len(gen_tensors) - len(gen_filtered)} incompatible tensors."
            )

            self._verify_encoder_weights(f"After loading checkpoint stage 0 from {checkpoint_path}")

        else:
            raise ValueError("Resume_type must be `resume` or `finetune`.")

        return checkpoint_path

    def _test_step(self, batch):
        raise NotImplementedError

    @torch.inference_mode()
    def _valid_epoch(self):
        r"""Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        epoch_sum_loss = 0.0
        return epoch_sum_loss

    def _inference(self):
        pass

    def save_checkpoint(self):
        """Override to add weight verification after saving."""
        # Call parent save_checkpoint
        super().save_checkpoint()
        
        self._verify_encoder_weights(f"After saving checkpoint at step {self.step}")
    
    def test_loop(self):
        return
