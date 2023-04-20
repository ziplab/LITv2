import torch
from models import build_model
from main import parse_option
import torch.onnx
import numpy as np

def main():
    _, config = parse_option()
    
    model = build_model(config).cuda()
    print(model)
    weights = torch.load(config.MODEL.RESUME, map_location='cpu')
    model.load_state_dict(weights['model'], strict=False)

    model.eval()
    BATCH_SIZE = 64
    dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224).cuda()
    model_save_path = f"./{config.MODEL.NAME}.onnx"
    print(f'output path : {model_save_path}')
    torch.onnx.export(model, dummy_input, model_save_path)


if __name__ == '__main__':
    main()