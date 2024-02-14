'''
    @author:
        - Leonardo Lo Schiavo
    @contributors
        - Gines Garcia-Aviles
        - Marco Gramaglia
        - Andres Garcia-Saavedra
    @affiliation:
        - IMDEA Networks institute
'''

import torch, os
import torch.nn as nn
import torch.nn.functional as F
from lpu_models import LPUModels



print("[INFO] Converting CPU model")
lpu_models = LPUModels()
torch_input = torch.randn(lpu_models.input_size)
onnx_program = torch.onnx.dynamo_export(lpu_models.predictor_time_cpu, torch_input)
onnx_program.save("onnx_model/predictor_time_cpu.onnx")

torch_input = torch.randn(lpu_models.input_size)
onnx_program = torch.onnx.dynamo_export(lpu_models.predictor_power_cpu, torch_input)
onnx_program.save("onnx_model/predictor_power_cpu.onnx")

print("[INFO] Converting GPU model")
torch_input = torch.randn(lpu_models.input_size)
onnx_program = torch.onnx.dynamo_export(lpu_models.predictor_time_gpu, torch_input)
onnx_program.save("onnx_model/predictor_time_gpu.onnx")

torch_input = torch.randn(lpu_models.input_size)
onnx_program = torch.onnx.dynamo_export(lpu_models.predictor_power_gpu, torch_input)
onnx_program.save("onnx_model/predictor_power_gpu.onnx")


