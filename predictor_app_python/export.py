import torch
from lpu_models import LPUModels

lpu_models = LPUModels()

torch_input = torch.randn(lpu_models.input_size)
onnx_program = torch.onnx.dynamo_export(lpu_models.predictor_time_cpu, torch_input)
onnx_program.save("../predictor_app_cplus/data/predictor_time_cpu.onnx")

torch_input = torch.randn(lpu_models.input_size)
onnx_program = torch.onnx.dynamo_export(lpu_models.predictor_time_gpu, torch_input)
onnx_program.save("../predictor_app_cplus/data/predictor_time_gpu.onnx")

torch_input = torch.randn(lpu_models.input_size)
onnx_program = torch.onnx.dynamo_export(lpu_models.predictor_power_cpu, torch_input)
onnx_program.save("../predictor_app_cplus/data/predictor_power_cpu.onnx")

torch_input = torch.randn(lpu_models.input_size)
onnx_program = torch.onnx.dynamo_export(lpu_models.predictor_power_gpu, torch_input)
onnx_program.save("../predictor_app_cplus/data/predictor_power_gpu.onnx")