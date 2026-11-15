import torch
import torch.nn as nn

class CNNEncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, padding_mode="zeros")
        self.l2_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, padding_mode="zeros")
        self.l3_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, padding_mode="zeros")
        self.l4_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, padding_mode="zeros")
        self.l1_act = nn.ReLU()
        self.l2_act = nn.ReLU()
        self.l3_act = nn.ReLU()
        self.l4_act = nn.ReLU()
    def forward(self, x):
        tem = self.l1_conv(x)
        skipper = self.l1_act(tem)
        tem = self.l2_conv(skipper)
        tem = self.l2_act(tem)
        tem = self.l3_conv(tem)
        tem = self.l3_act(tem)
        tem = self.l4_conv(tem)
        res = self.l4_act(tem)
        return res + skipper

class CNNDecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_conv = nn.Conv1d(2, 1, kernel_size=5, padding=2, padding_mode="zeros")
        self.l2_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, padding_mode="zeros")
        self.l3_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, padding_mode="zeros")
        self.l4_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, padding_mode="zeros")
        self.l1_act = nn.ReLU()
        self.l2_act = nn.ReLU()
        self.l3_act = nn.ReLU()
        self.l4_act = nn.ReLU()
    def forward(self, hidden, x):
        tem = self.l1_conv(torch.cat([hidden, x], dim=-2))
        skipper = self.l1_act(tem)
        tem = self.l2_conv(skipper)
        tem = self.l2_act(tem)
        tem = self.l3_conv(tem)
        tem = self.l3_act(tem)
        tem = self.l4_conv(tem)
        res = self.l4_act(tem)
        return res + skipper

class AEncoder1DUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode_block_1 = CNNEncoderBlock()
        self.encode_block_2 = CNNEncoderBlock()
        self.decode_block_1 = CNNDecoderBlock()
        self.decode_block_2 = CNNDecoderBlock()
        self.downscaled_l_1 = nn.MaxPool1d(3, 2)
        self.downscaled_l_2 = nn.MaxPool1d(3, 2)
        self.upscale_l_1 = nn.ConvTranspose1d(1, 1, 4, 2)
        self.upscale_l_2 = nn.ConvTranspose1d(1, 1, 3, 2)
        self.l1_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, padding_mode="zeros")
        self.l2_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, padding_mode="zeros")
        self.l3_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, padding_mode="zeros")
        self.l1_act = nn.ReLU()
        self.l2_act = nn.ReLU()
        self.l3_act = nn.ReLU()
        self.l_output_conv = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, x):
        hidden_rep_1 = self.encode_block_1(x)
        downscaled_hidden_rep_1 = self.downscaled_l_1(hidden_rep_1)
        hidden_rep_2 = self.encode_block_2(downscaled_hidden_rep_1)
        downscaled_hidden_rep_2 = self.downscaled_l_2(hidden_rep_2)
        hidden_low = self.l1_conv(downscaled_hidden_rep_2)
        hidden_low = self.l1_act(hidden_low)
        hidden_low = self.l2_conv(hidden_low)
        hidden_low = self.l2_act(hidden_low)
        hidden_low = self.l3_conv(hidden_low)
        hidden_low = self.l3_act(hidden_low)
        upscaled_hidden_rep_2 = self.upscale_l_2(hidden_low)
        reconstruction_output_2 = self.decode_block_2(hidden_rep_2, upscaled_hidden_rep_2)
        upscaled_hidden_rep_1 = self.upscale_l_1(reconstruction_output_2)
        reconstruction_output_1 = self.decode_block_1(hidden_rep_1, upscaled_hidden_rep_1)
        res = self.l_output_conv(reconstruction_output_1)
        return res
