import dualcodec
import torch

w2v_path = "./w2v-bert-2.0" # your downloaded path
dualcodec_model_path = "./output_checkpoints/dualcodec_25hz_16384_1024/checkpoint/epoch-0000_step-0002800_loss-135.952698-dualcodec_25hz_16384_1024/" # your downloaded path
model_id = "25hz_v1" # select from available Model_IDs, "12hz_v1" or "25hz_v1"

dualcodec_model = dualcodec.get_model(model_id, dualcodec_model_path)
dualcodec_inference = dualcodec.Inference(dualcodec_model=dualcodec_model, dualcodec_path=dualcodec_model_path, w2v_path=w2v_path, device="cuda")

# do inference for your wav
import torchaudio
audio, sr = torchaudio.load("sam.wav")
# resample to 24kHz
audio = torchaudio.functional.resample(audio, sr, 24000)
audio = audio.reshape(1,1,-1)
audio = audio.to("cuda")
# extract codes, for example, using 8 quantizers here:
semantic_codes, acoustic_codes = dualcodec_inference.encode(audio, n_quantizers=8)
# semantic_codes shape: torch.Size([1, 1, T])
# acoustic_codes shape: torch.Size([1, n_quantizers-1, T])


all_codes = torch.cat([semantic_codes, acoustic_codes], dim=1)
all_codes = all_codes.squeeze(0)
#shape is N,T

for idx, code in enumerate(all_codes):
    unique_vals = torch.unique(code)
    num_unique = unique_vals.numel()
    #print max and min
    print(f"Layer {idx}: {num_unique} unique codes, max: {torch.max(code)}, min: {torch.min(code)}")
    if num_unique < 5:
        print(f"Layer {idx} few codes → {unique_vals.tolist()}")
# produce output audio. If `acoustic_codes=None` is passed, will decode only semantic codes (RVQ-1)
out_audio = dualcodec_inference.decode(semantic_codes, acoustic_codes)

# save output audio
torchaudio.save("out.wav", out_audio.cpu().squeeze(0), 24000)