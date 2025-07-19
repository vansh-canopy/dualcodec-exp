import torch
import torchaudio
import torch.nn.functional as F
import warnings
from contextlib import nullcontext
from easydict import EasyDict as edict

from transformers import WhisperFeatureExtractor, WhisperConfig
from dualcodec.infer.dualcodec.causal_whisper_wrapper import CausalWhisperModel
from cached_path import cached_path



def _build_whisper_semantic_model(
    device="cuda",
    whisper_config_path="openai/whisper-base",
    whisper_model_path="vanshjjw/whisper-stream-lookahead-3",   
    **kwargs,
):
    if not torch.cuda.is_available():
        warnings.warn("CUDA is not available, running on CPU.")
        device = "cpu"

    whisper_conf = WhisperConfig.from_pretrained(whisper_config_path)
    model = CausalWhisperModel.from_pretrained(whisper_model_path, config=whisper_conf)
    model = model.eval().to(device)  # type: ignore

    feat_extractor = WhisperFeatureExtractor.from_pretrained(whisper_config_path)

    # Whisper-base encoder has 32 layers → choose last hidden state (31)
    output_idx = 31

    return edict(
        {
            "semantic_model": model,
            "layer_idx": output_idx,  # retained for compatibility
            "output_idx": output_idx,
            "feature_extractor": feat_extractor,
            "skip_semantic_normalize": True,  # no fixed stats
        }
    )

def pad_to_length(x, length, pad_value=0):
    current_length = x.shape[-1]
    if length > current_length:
        pad_amount = length - current_length
        x_padded = F.pad(x, (0, pad_amount), value=pad_value)
    else:
        x_padded = x[..., :length]
    return x_padded


class InferenceWhisper:
    """DualCodec inference wrapper that uses Whisper-base for semantic features."""

    def __init__(
        self,
        dualcodec_model,
        dualcodec_path="hf://amphion/dualcodec",
        whisper_path="openai/whisper-base",
        device="cuda",
        autocast=True,
        **kwargs,
    ) -> None:
        dualcodec_path = str(cached_path(dualcodec_path))  # ensure plain str for torch load
        
        if not torch.cuda.is_available():
            warnings.warn("CUDA is not available, running on CPU.")
            device = "cpu"

        self.semantic_cfg = _build_whisper_semantic_model(
            device=device, whisper_config_path=whisper_path
        )

        self.model = dualcodec_model.to(device).eval()

        # Move tensors / sub-modules to device
        for k, v in self.semantic_cfg.items():
            if isinstance(v, (torch.nn.Module, torch.Tensor)):
                self.semantic_cfg[k] = v.to(device)

        # Ensure causal mask inside whisper encoder (if present) is on the same device
        whisper_model = self.semantic_cfg["semantic_model"]
        if hasattr(whisper_model, "encoder") and hasattr(whisper_model.encoder, "causal_mask"):
            whisper_model.encoder.causal_mask = whisper_model.encoder._create_lookahead_mask(1500, 3)
            whisper_model.encoder.causal_mask = whisper_model.encoder.causal_mask.to(device)

        self.device = device
        self.autocast = autocast

    
    @torch.no_grad()
    def encode(self, audio: torch.Tensor, n_quantizers: int = 8):
        """Encode audio to (semantic_codes, acoustic_codes)."""
        audio_16k = torchaudio.functional.resample(audio, 24000, 16000)

        # WhisperFeatureExtractor expects 1-D waveform (time,) or list of such.
        if audio_16k.dim() == 3:
            audio_input = audio_16k[0, 0].cpu()
        elif audio_16k.dim() == 2:
            audio_input = audio_16k[0].cpu()
        else:
            audio_input = audio_16k.cpu()
        
        print("audio_16k.shape", audio_16k.shape)    
        print("audio_input.shape", audio_input.shape)        

        feat_extractor = self.semantic_cfg["feature_extractor"]
        inputs = feat_extractor(audio_input, sampling_rate=16000, return_tensors="pt")
        
        input_features = inputs["input_features"][0]
        
        print("input_features.shape", input_features.shape)
        
        input_features = input_features.unsqueeze(0).to(self.device)
        audio = audio.to(self.device)

        with torch.autocast(device_type=self.device, dtype=torch.float16):
            feat = self._extract_semantic_code(input_features).transpose(1, 2)
            
            print("feat.shape", feat.shape)
            
            feat = torch.nn.functional.avg_pool1d(
                feat,
                self.model.semantic_downsample_factor,
                self.model.semantic_downsample_factor,
            )
            
            print("feat.shape after avg_pool1d", feat.shape)

        ctx = (
            torch.autocast(device_type=self.device, dtype=torch.float16)
            if self.autocast
            else nullcontext()
        )
        with ctx:
            semantic_codes, acoustic_codes = self.model.encode(
                audio, num_quantizers=n_quantizers, semantic_repr=feat
            )

        return semantic_codes, acoustic_codes

    @torch.no_grad()
    def decode_from_codes(self, semantic_codes, acoustic_codes):
        audio = self.model.decode_from_codes(semantic_codes, acoustic_codes).to(torch.float32)
        return audio

    @torch.no_grad()
    def decode(self, semantic_codes, acoustic_codes):
        """Alias for parity with original Inference class."""
        return self.decode_from_codes(semantic_codes, acoustic_codes)

    @torch.no_grad()
    def _extract_semantic_code(self, input_features):
        # Run only the Whisper encoder to obtain hidden states (no decoder needed)
        encoder_out = self.semantic_cfg["semantic_model"].encoder(input_features=input_features)
        feat = encoder_out.last_hidden_state  # (B, T, 512)
       # Return raw 512-dim features; DualCodec will map to 1024 internally
        return feat



def infer(audio, inference_wrapper: "InferenceWhisper", num_quantizers: int = 8):
    """Quick convenience wrapper mirroring inference_with_semantic.infer."""
    audio = audio.reshape(1, 1, -1).cpu()
    semantic_codes, acoustic_codes = inference_wrapper.encode(audio, n_quantizers=num_quantizers)
    out_audio = inference_wrapper.model.decode_from_codes(semantic_codes, acoustic_codes)
    out_audio = pad_to_length(out_audio, audio.shape[-1])
    return out_audio, (semantic_codes, acoustic_codes) 