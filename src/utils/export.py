import torch
import os
from src.models.sign_model import ASLClassifier

def export_models(model_path="models/best_model.pth", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    
    model = ASLClassifier()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # 1. Dynamic Quantization (Optimizes for CPU inference)
    print("Applying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    quantized_path = os.path.join(output_dir, "model_quantized.pth")
    torch.save(quantized_model.state_dict(), quantized_path)
    print(f"Saved quantized model to {quantized_path}")
    
    # 2. ONNX Export (For cross-platform deployment)
    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, 63)
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Saved ONNX model to {onnx_path}")

if __name__ == "__main__":
    export_models()
