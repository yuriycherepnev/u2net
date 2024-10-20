import torch
import torch.onnx
from model.u2net import U2NET

import io
import numpy as np
import torch.onnx
from model import U2NET


def load_model(model_path, model_class):
    torch_model = U2NET(3, 1)
    torch_model.load_state_dict(torch.load(model_path), strict=False)
    torch_model.eval()

    return torch_model

def convert_to_onnx(model, output_path):

    dummy_input = torch.randn(1, 3, 320, 320)

    torch.onnx.export(model, dummy_input, output_path, opset_version=9,
                          dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                                        'output': {0: 'batch_size', 2: 'height', 3: 'width'}})
    print(f"success {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="conversion PyTorch to ONNX")
    parser.add_argument('--model-path', type=str, required=True, help='path to .pth file')
    parser.add_argument('--output-path', type=str, required=True, help='save ONNX file')

    args = parser.parse_args()
    model = load_model(args.model_path, U2NET)
    convert_to_onnx(model, args.output_path)
