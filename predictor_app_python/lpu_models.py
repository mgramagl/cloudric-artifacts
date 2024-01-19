'''
    @author:
        - Leonardo Lo Schiavo
    @affiliation:
        - IMDEA Networks institute
'''
import torch
import numpy as np

from predictor import Predictor

class LPUModels:

    def __init__(self):
        # Dataset parameters
        self.max_snr = 30.0
        self.max_mcs = 27
        self.max_total_bits = 295680
        self.max_prbs = 250

        # LPU models parameters
        self.input_size = 3
        self.output_size = 1
        self.hidden_size = 128

        # Import max and min values for the inputs (only GPU power model)
        self.max_gpu_power = np.load('data/max_power_gpu.npy')
        self.min_gpu_power = np.load('data/min_power_gpu.npy')

        # Load the model weights and set the model to inference mode
        self.predictor_time_cpu = Predictor(self.input_size, self.output_size, self.hidden_size)
        self.predictor_time_cpu.load_state_dict(torch.load('data/predictor_time_cpu.pyt', map_location="cpu"))
        self.predictor_time_cpu.eval()
        self.predictor_time_gpu = Predictor(self.input_size, self.output_size, self.hidden_size)
        self.predictor_time_gpu.load_state_dict(torch.load('data/predictor_time_gpu.pyt', map_location="cpu"))
        self.predictor_time_gpu.eval()
        self.predictor_power_cpu = Predictor(self.input_size, self.output_size, self.hidden_size)
        self.predictor_power_cpu.load_state_dict(torch.load('data/predictor_power_cpu.pyt', map_location="cpu"))
        self.predictor_power_cpu.eval()
        self.predictor_power_gpu = Predictor(self.input_size, self.output_size, self.hidden_size)
        self.predictor_power_gpu.load_state_dict(torch.load('data/predictor_power_gpu.pyt', map_location="cpu"))
        self.predictor_power_gpu.eval()


    def estimate_service_time(self, snr, mcs, prbs, total_bits):
        # Normalize the inputs and format them for PyTorch
        power_inputs = []
        power_inputs.append(snr / self.max_snr)
        power_inputs.append(mcs / self.max_mcs)
        power_inputs.append(prbs / self.max_prbs)
        power_inputs = torch.Tensor(power_inputs)

        time_inputs = []
        time_inputs.append(snr / self.max_snr)
        time_inputs.append(mcs / self.max_mcs)
        time_inputs.append(total_bits / self.max_total_bits)
        time_inputs = torch.Tensor(time_inputs)
        
        # Run the model in inference mode
        power_cpu = self.predictor_power_cpu(power_inputs).detach().numpy()[0]
        time_cpu = float(self.predictor_time_cpu(time_inputs).detach().numpy()[0])
        energy_cpu = float(power_cpu * time_cpu)
        
        power_gpu = self.predictor_power_gpu(power_inputs).detach().numpy()[0] * (self.max_gpu_power - self.min_gpu_power) + self.min_gpu_power
        time_gpu = float(self.predictor_time_gpu(time_inputs).detach().numpy()[0])
        energy_gpu = float(power_gpu * time_gpu)

        return time_cpu, energy_cpu, time_gpu, energy_gpu