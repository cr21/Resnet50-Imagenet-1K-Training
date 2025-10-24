import os
from pathlib import Path
import logging
import torch
from utils import DeploymentConfig
import torchvision
from model import ResNet50Wrapper
import argparse



def main(cfg: DeploymentConfig) -> None:
    # Initialize Model
    
    # create instance of ResNet50Wrapper with the same number of classes as training
    model = ResNet50Wrapper(num_classes=1000)
    
    # load checkpoint from checkpoint path
    if os.path.exists(cfg.ckpt_path):
        checkpoint = torch.load(cfg.ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["model"])
    else:
        raise ValueError(f"Checkpoint file not found at {cfg.ckpt_path}")

    
    # Set model to eval mode and device
    device = cfg.device # deploy on cpu on Huggingface space
    model = model.to(device)
    model.eval()
    img_w = cfg.IMG_W or 224
    img_h = cfg.IMG_H or 224
    # Create example input
    example_input = torch.randn(1, 3, img_h, img_w).to(device)  # Move input to same device as model
    
    # Trace the model
    print(f"Tracing model on device: {device}")
    traced_model = torch.jit.trace(model, example_input)
    print(traced_model.graph)
    # Move traced model to CPU before saving
    traced_model = traced_model.to(device)
    
    # Create output directory if it doesn't exist
    output_dir = Path(f"traced_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the traced model
    output_path = output_dir / "resnet50_imagenet_1k_model.pt"
    torch.jit.save(traced_model, output_path)
    print(f"Traced model saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    args = parser.parse_args()
    config = DeploymentConfig()
    config.ckpt_path = args.ckpt_path
    main(config)