import torch
from denoisingSeqAutoEncoder import AEncoder1DUNet


if __name__ == "__main__":
    input_data = torch.randn((4,1,188))
    model = AEncoder1DUNet()
    res = model(input_data)
    print(f"data size: {input_data.size()}\nresult size: {res.size()}")