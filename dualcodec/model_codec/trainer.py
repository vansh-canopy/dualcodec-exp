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

USE_HINGE_LOSS = False
from audiotools import AudioSignal
from dualcodec.infer.dualcodec.causal_whisper_wrapper import CausalWhisperModel


class Trainer(BaseTrainer):
    """Trainer"""

    def __init__(self, args=None, cfg=None, **kwargs):
        """
            Initializes the model with the given arguments and configuration.

        Args:
            args (argparse.Namespace, optional): Arguments to be passed on to the model. Defaults to None.
            cfg (dict, optional): Configuration dictionary containing parameters for the model. Defaults to None.
        """
        super().__init__(args, cfg)
        torch.backends.cudnn.benchmark = True
        # Flag to indicate this trainer handles its own backward passes
        self.codec_trainer_handles_backward = True

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


    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def _extract_semantic_code(self, input_features, attention_mask=None):
        sem_mod = self.cfg.semantic_model["model"]

        if isinstance(sem_mod, CausalWhisperModel):
            features = sem_mod.encoder(input_features=input_features).last_hidden_state
        else:
            vq_emb = sem_mod(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            features = vq_emb.hidden_states[self.cfg.semantic_model["output_idx"]]

            if not self.cfg.semantic_model.get("skip_semantic_normalize", False):
                features = (
                    features - self.cfg.semantic_model["mean"]
                ) / self.cfg.semantic_model["std"]

        return features

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

    def _train_step(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.accelerator.device)

        x_wav, audio_lengths = batch["speech"], batch["speech_lens"]

        x_wav = x_wav.float()[:, None, :]

        if self.cfg.semantic_vq:
            input_features = batch["input_features"]
            attention_mask = batch["attention_mask"]
            
            features = self._extract_semantic_code(
                input_features, attention_mask
            ).transpose(1, 2)
            
            # Wave 2 Vec produces the right number of latents (50/s)
            # So does whisper but because openai is a bitch, they pad the latents upto 1500
            # quick hack to drop padded whisper latents
            
            sr = 24000 
            audio_len_in_s = audio_lengths[0].item() / sr
            num_latents_to_keep = int(50 * audio_len_in_s)
            features = features[:,:,:num_latents_to_keep]
            
            features = torch.nn.functional.avg_pool1d(
                features,
                self.model_module.semantic_downsample_factor,
                self.model_module.semantic_downsample_factor,
            )
            
            out_dict, semantic_edict = self.model(
                x_wav,
                semantic_repr=features,
                bypass_quantize_rate=0.125,
                possibly_no_quantizer=False,
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
            adv_loss, feat_loss = self.gan_loss.generator_hinge_loss(
                generator_out, x_wav
            )
        else:
            adv_loss, feat_loss = self.gan_loss.generator_loss(generator_out, x_wav)
        spec_loss = self.spec_loss(
            AudioSignal(x_wav, 24000), AudioSignal(generator_out, 24000)
        )
        
        total_loss = (
            0.25 * commitment_loss 
            + 2.0 * adv_loss 
            + 4.0 * feat_loss
            + 15.0 * spec_loss 
            + 1.0 * codebook_loss
        )
        
        # ---------- Generator training ----------------
        
        if semantic_edict:
            distill_loss = F.mse_loss(features, semantic_edict["x"])
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
            features = self._extract_semantic_code(
                input_features, attention_mask
            ).transpose(1, 2)
            features = torch.nn.functional.avg_pool1d(
                features, self.semantic_downsample_factor, self.semantic_downsample_factor
            )
            distill_loss = F.mse_loss(
                features, first_layer_quantized[..., : features.shape[-1]]
            )
            total_loss += distill_loss * self.cfg.lambda_distill_loss
        else:
            distill_loss = 0.0

        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        metrics.update(
            {
                "commitment_loss": commitment_loss,
                "spec_loss": spec_loss,
                "feat_loss": feat_loss,
                "adv_g_loss": adv_loss,
                "total_loss": total_loss,
                "Train/Batch Size": x_wav.shape[0],
                "disc_loss": disc_loss.item(),
                "distill_loss": distill_loss,
            }
        )

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
                self.logger.info("Resume from {}".format(checkpoint_path))
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

        elif resume_type == "finetune":
            # Load only the model weights
            import safetensors.torch

            safetensors.torch.load_model(
                self.accelerator.unwrap_model(self.model),
                os.path.join(
                    checkpoint_path, self.args.model_1_name
                ),  # location of "model_1.safetensors"
            )
            safetensors.torch.load_model(
                self.accelerator.unwrap_model(self.discriminator),
                os.path.join(
                    checkpoint_path, self.args.model_2_name
                ),  # location of "model_2.safetensors"
            )
            self.logger.info("Loaded model weights for finetune.")

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

    def test_loop(self):
        return
