
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

# cuBLAS determinism (needed for some GPU configs)
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

MODEL_ID = "12hz_v1"    
DEVICE = "cuda:3"

model_base = dualcodec.get_model(MODEL_ID)
model_base.eval()
inference_base = dualcodec.Inference(dualcodec_model=model_base, device=DEVICE, autocast=False)

# name_2 = "exp_3_step_74000_last_10_decay_0.99.safetensors"

# model_2 = dualcodec.get_model(MODEL_ID, path, name=name_2, strict=False)
# model_2.eval()
# inference_2 = dualcodec.Inference(dualcodec_model=model_2, device=DEVICE, autocast=False)

path = "/home/vansh/dualcodec-exp/averaged_models"
name_frozen = "exp_1_step_76600_last_10_decay_0.99.safetensors"

model_enc_quan_frozen = dualcodec.get_model(MODEL_ID, path, name=name_frozen, strict=False)
model_enc_quan_frozen.eval()
inference_enc_quan_frozen = dualcodec.Inference(dualcodec_model=model_enc_quan_frozen, device=DEVICE, autocast=False)


def compare_state_dicts(sd1, sd2, module_name=""):
    all_close = True
    for (k1, v1), (k2, v2) in zip(sd1.items(), sd2.items()):
        if k1 != k2:
            print(f"[{module_name}] Key mismatch: {k1} vs {k2}")
            continue
        
        diff = torch.abs(v1 - v2)
        maximum = torch.max(diff)
        print(f"[{module_name}] {k1} difference: {maximum}")
            
    if all_close:
        print("All weights are identical")

# compare_state_dicts(model_base.dac.encoder.state_dict(), model_enc_quan_frozen.dac.encoder.state_dict(), "Encoder")
# compare_state_dicts(model_base.dac.quantizer.state_dict(), model_enc_quan_frozen.dac.quantizer.state_dict(), "Quantizer")
# compare_state_dicts(model_base.convnext_encoder.state_dict(), model_enc_quan_frozen.convnext_encoder.state_dict(), "ConvNextEncoder")
# compare_state_dicts(model_base.semantic_vq.state_dict(), model_enc_quan_frozen.semantic_vq.state_dict(), "SemanticQuantizer")

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
        
        if wav_paths[i] != "audio_samples/luna.wav":
            continue
        
        audio = audio.reshape(1,1,-1)
        audio = audio.to(DEVICE)
        
        sem1, acu1 = inference_enc_quan_frozen.encode(audio, n_quantizers=3)
        
        audio_decoded_1 = inference_enc_quan_frozen.decode(sem1, acu1)
        audio_decoded_1 = audio_decoded_1.squeeze(0).cpu()
        torchaudio.save(f"audio_decoded_1.wav", audio_decoded_1, 24000)
        
        audio_decoded_2 = inference_base.decode(sem1, acu1)
        audio_decoded_2 = audio_decoded_2.squeeze(0).cpu()
        torchaudio.save(f"audio_decoded_2.wav", audio_decoded_2, 24000)
        
        # enc1 = inference_base.model.dac.encoder(audio)
        # enc2 = inference_enc_quan_frozen.model.dac.encoder(audio)
        # print(f"audio name: {wav_paths[i]}")
        # print(f"Encoder latents difference: {enc1 - enc2}")

        # sem1, acu1 = inference_base.encode(audio, n_quantizers=2)
        # sem2, acu2 = inference_enc_quan_frozen.encode(audio, n_quantizers=2)
        
        # print(f"acoustic codes shape: {acu1.shape} and {acu2.shape}")
        
        # diff_first_code = acu1[:,0,:] - acu2[:,0,:]
        # print(f"first code difference: {diff_first_code}")
        
        # diff_second_code = acu1[:,1,:] - acu2[:,1,:]
        # print(f"second code difference: {diff_second_code}")
        

        
    
    
