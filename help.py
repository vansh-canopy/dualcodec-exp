
import random
import numpy as np
import torch
import torchaudio
import dualcodec

# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)

# # cuBLAS determinism (needed for some GPU configs)
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

MODEL_ID = "12hz_v1"    

DEVICE = "cuda:3"

model = dualcodec.get_model(MODEL_ID)
model.eval()

inference = dualcodec.Inference(dualcodec_model=model, device=DEVICE, autocast=False)

path = "/home/vansh/dualcodec-exp/output_checkpoints_rough/dualcodec_25hzv1_finetune/checkpoint/epoch-0022_step-0114800_loss-71.511154-dualcodec_25hzv1_finetune"

model_2 = dualcodec.get_model(MODEL_ID, path, name="model.safetensors")
model_2.eval()
inference_2 = dualcodec.Inference(dualcodec_model=model_2, device=DEVICE, autocast=False)

# Load and resample audio to 24 kHz (model's native rate)
AUDIO_PATH = "audio_samples/tara.wav"
audio, sr = torchaudio.load(AUDIO_PATH)
audio = torchaudio.functional.resample(audio, sr, 24000)
audio = audio.reshape(1,1,-1)
audio = audio.to(DEVICE)


with torch.no_grad():
    sem1, acu1 = inference.encode(audio, n_quantizers=8)
    sem2, acu2 = inference_2.encode(audio, n_quantizers=8)

    print(f"Semantic codes difference: {sem1 - sem2}")
    print(f"Acoustic codes difference: {acu1 - acu2}")

    enc1 = inference.model.dac.encoder(audio)
    enc2 = inference_2.model.dac.encoder(audio)
    print(f"Encoder latents difference: {enc1 - enc2}")
    
    
