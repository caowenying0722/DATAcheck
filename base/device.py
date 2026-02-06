import torch


class Device:
    device_name: str
    device: torch.device

    def __init__(self):
        if torch.cuda.is_available():
            self.device_name = "cuda"
        elif torch.mps.is_available():
            self.device_name = "mps"
        else:
            self.device_name = "cpu"
        self.device = torch.device(self.device_name)


CpuDevice = torch.device("cpu")
DefaultDevice = Device().device
DefaultDevice = CpuDevice
