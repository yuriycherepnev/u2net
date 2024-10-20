import torch
import torch.onnx
from model import U2NET

def convert(model_path, output_path):
    torch_model = U2NET(3, 1)
    torch_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)  # Добавлен map_location
    torch_model.eval()

    x = torch.randn(1, 3, 320, 320, requires_grad=True)
    torch_out = torch_model(x)

    torch.onnx.export(
        torch_model,
        x,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"success {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="conversion PyTorch to ONNX")
    parser.add_argument('--model-path', type=str, required=True, help='path to .pth file')
    parser.add_argument('--output-path', type=str, required=True, help='save ONNX file')

    args = parser.parse_args()
    convert(args.model_path, args.output_path)
