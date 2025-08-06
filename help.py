
import random
import numpy as np
import torch
import torchaudio
import dualcodec
import pathlib

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

model_1 = dualcodec.get_model(MODEL_ID)
model_1.eval()
inference = dualcodec.Inference(dualcodec_model=model_1, device=DEVICE, autocast=False)

path = "/home/vansh/dualcodec-exp/averaged_models"
name = "exp_1_step_76600_last_10_decay_0.99.safetensors"

model_2 = dualcodec.get_model(MODEL_ID, path, name=name, strict=False)
model_2.eval()
inference_2 = dualcodec.Inference(dualcodec_model=model_2, device=DEVICE, autocast=False)


SAMPLES_DIR = "/home/vansh/dualcodec-exp/audio_samples"

wav_paths = sorted(str(p) for p in pathlib.Path(SAMPLES_DIR).glob("*.wav"))
if not wav_paths:
    raise FileNotFoundError("No .wav found in " + SAMPLES_DIR)

samples = []
for p in wav_paths:
    sample, sr = torchaudio.load(p)
    if sr != 24000:
        sample = torchaudio.transforms.Resample(sr, 24000)(sample)
    samples.append(sample)


with torch.no_grad():
    for i,audio in enumerate(samples):
        audio = audio.reshape(1,1,-1)
        audio = audio.to(DEVICE)
        
        enc1 = inference.model.dac.encoder(audio)
        enc2 = inference_2.model.dac.encoder(audio)
        print(f"audio name: {wav_paths[i]}")
        print(f"Encoder latents difference: {enc1 - enc2}")

        # sem1, acu1 = inference.encode(audio, n_quantizers=2)
        # sem2, acu2 = inference.encode(audio, n_quantizers=2)
        # print(f"Semantic codes difference: {sem1 - sem2}")
        # print(f"Acoustic codes difference: {acu1 - acu2}")

        
    
    
