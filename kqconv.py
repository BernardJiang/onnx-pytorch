import torch
import torch.nn as nn
# from ..intrinsic import ConvAct2d
# @torch.fx.wrap
class KQConv2d(nn.Conv2d):
    r"""
    A Conv2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv2d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
    for documentation.

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Conv2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', qconfig=None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        assert qconfig, 'qconfig must be provided for QAT module'
        # self.qconfig = qconfig
        # self.weight_fake_quant = qconfig.weight()

    def forward(self, input):
        return self._conv_forward(input + 1., self.weight, self.bias)

