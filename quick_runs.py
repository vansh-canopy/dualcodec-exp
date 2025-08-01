
from dualcodec.model_codec import dac_layers
from dualcodec.model_codec.dac_model import DAC
import torch

inputs_res = torch.randn(1, 768, 10)

dac = DAC(decoder_rates=[2,8,6,5,4]).eval()
resUnits = dac.decoder.model[1].block[2] # type: ignore
print(resUnits)

outputs = resUnits(inputs_res)  # type: ignore
print(outputs.shape)


print(inputs_res[0][0])
print(outputs[0][0])

