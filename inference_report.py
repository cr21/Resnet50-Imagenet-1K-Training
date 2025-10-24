import torch
import torch.nn as nn
from model import ResNet50Wrapper
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
VAL_DATA_PATH = '/mnt/data/imagenet/val'
CHECKPOINT_PATH = '/home/ubuntu/Resnet50-Imagenet-1K-Training/checkpoint-epoch-49-train_acc-75.65-test_acc-75.32-train_acc5-91.12-test_acc5-cle92.59.pth'
LABELS_PATH = '/home/ubuntu/Resnet50-Imagenet-1K-Training/imagenet_labels.txt'

def load_imagenet_labels(labels_path):
    """Load ImageNet class labels from file."""
    labels = []
    with open(labels_path, 'r') as f:
        for line in f:
            # Format: "n01440764 tench, Tinca tinca"
            parts = line.strip().split(' ', 1)
            if len(parts) > 1:
                labels.append(parts[1])
            else:
                labels.append(parts[0])
    return labels

def load_model(checkpoint_path, num_classes=1000, device='cpu'):
    """Load ResNet50 model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Initialize model
    model = ResNet50Wrapper(num_classes=num_classes, use_checkpoint=False)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load state dict (handle DDP wrapper if present)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DDP)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model

def get_val_dataloader(batch_size=32, num_workers=4):
    """Create validation data loader for ImageNet."""
    val_transformation = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=VAL_DATA_PATH,
        transform=val_transformation
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return val_loader, val_dataset

def denormalize_image(image):
    """Denormalize ImageNet image for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return image * std + mean

def evaluate_model(model, dataloader, device='cpu'):
    """Evaluate model on validation set."""
    model.eval()
    correct = 0
    correct_top5 = 0
    total = 0
    
    print("Evaluating model on validation set...")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Top-1 accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            
            # Top-5 accuracy
            _, pred_top5 = outputs.topk(5, 1, largest=True, sorted=True)
            correct_top5 += pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).sum().item()
            
            total += labels.size(0)
    
    top1_accuracy = 100 * correct / total
    top5_accuracy = 100 * correct_top5 / total
    
    print(f"\n{'='*50}")
    print(f"Validation Results:")
    print(f"{'='*50}")
    print(f"Total samples: {total}")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
    print(f"{'='*50}\n")
    
    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'total_samples': total,
        'correct_top1': correct,
        'correct_top5': correct_top5
    }

def get_misclassified_images(model, dataloader, classes, device='cpu', max_images=20, use_gradcam=True):
    """Get misclassified images with their predictions."""
    model.eval()
    
    error_images = []
    error_images_gradcam = []
    error_labels = []
    error_preds = []
    error_probs = []
    error_paths = []
    
    print(f"Finding misclassified images (max {max_images})...")
    
    count = 0
    for images, labels in tqdm(dataloader, desc="Finding misclassifications"):
        if count >= max_images:
            break
            
        images = images.to(device)
        labels = labels.to(device)
        
        # First pass: find misclassifications (no gradients needed)
        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=-1)
            pred_probs, pred_classes = torch.max(probs, dim=1)
            
            # Find misclassified images in this batch
            misclassified_mask = (pred_classes != labels)
        
        for idx in range(len(images)):
            if count >= max_images:
                break
                
            if misclassified_mask[idx]:
                error_images.append(images[idx].cpu())
                error_labels.append(labels[idx].item())
                error_preds.append(pred_classes[idx].item())
                error_probs.append(pred_probs[idx].item())
                
                # Generate GradCAM if requested (OUTSIDE no_grad context)
                if use_gradcam:
                    try:
                        # Create a new tensor with gradient enabled
                        img_tensor = images[idx].unsqueeze(0).clone().detach()
                        img_tensor.requires_grad = True
                        
                        target_layers = [model.model.layer4[-1]]
                        cam = GradCAM(model=model, target_layers=target_layers)
                        grayscale_cam = cam(input_tensor=img_tensor, targets=None)
                        grayscale_cam = grayscale_cam[0, :]
                        
                        # Denormalize for visualization
                        img_np = denormalize_image(images[idx].permute(1, 2, 0).cpu().numpy())
                        img_np = np.clip(img_np, 0, 1)
                        
                        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True, image_weight=0.5)
                        error_images_gradcam.append(visualization)
                    except Exception as e:
                        print(f"GradCAM failed: {e}")
                        error_images_gradcam.append(None)
                
                count += 1
    
    print(f"Found {len(error_images)} misclassified images")
    
    return {
        'images': error_images,
        'gradcam_images': error_images_gradcam if use_gradcam else None,
        'labels': error_labels,
        'predictions': error_preds,
        'probabilities': error_probs
    }

def plot_misclassified_images(misclassified_data, classes, save_path='misclassified_report.png', use_gradcam=True):
    """Plot and save misclassified images."""
    images = misclassified_data['images']
    labels = misclassified_data['labels']
    preds = misclassified_data['predictions']
    probs = misclassified_data['probabilities']
    gradcam_images = misclassified_data['gradcam_images']
    
    num_images = len(images)
    
    if num_images == 0:
        print("No misclassified images to plot!")
        return
    
    # Calculate grid size
    cols = 5
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        if idx < num_images:
            if use_gradcam and gradcam_images and gradcam_images[idx] is not None:
                ax.imshow(gradcam_images[idx])
            else:
                img_np = denormalize_image(images[idx].permute(1, 2, 0).numpy())
                img_np = np.clip(img_np, 0, 1)
                ax.imshow(img_np)
            
            true_label = classes[labels[idx]][:50]  # Truncate long labels
            pred_label = classes[preds[idx]][:50]
            prob = probs[idx]
            
            ax.set_title(f'True: {true_label}\nPred: {pred_label} ({prob:.2f})', fontsize=8)
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Misclassified images plot saved to {save_path}")
    plt.close()

def save_individual_misclassified_images(misclassified_data, classes, output_dir='misclassified_images'):
    """Save individual misclassified images."""
    os.makedirs(output_dir, exist_ok=True)
    
    images = misclassified_data['images']
    labels = misclassified_data['labels']
    preds = misclassified_data['predictions']
    probs = misclassified_data['probabilities']
    
    print(f"Saving individual images to {output_dir}...")
    
    for idx in range(len(images)):
        img_np = denormalize_image(images[idx].permute(1, 2, 0).numpy())
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        
        true_label = classes[labels[idx]].split(',')[0].replace(' ', '_')
        pred_label = classes[preds[idx]].split(',')[0].replace(' ', '_')
        prob = probs[idx]
        
        filename = f"misclassified_{idx:03d}_true_{true_label}_pred_{pred_label}_{prob:.2f}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        img_pil = Image.fromarray(img_np)
        img_pil.save(filepath)
    
    print(f"Saved {len(images)} images to {output_dir}")

def main():
    """Main inference and reporting function."""
    print("="*70)
    print("ResNet50 ImageNet-1K Inference Report")
    print("="*70)
    
    # Load ImageNet labels
    print(f"\nLoading ImageNet labels from {LABELS_PATH}")
    classes = load_imagenet_labels(LABELS_PATH)
    print(f"Loaded {len(classes)} class labels")
    
    # Load model
    model = load_model(CHECKPOINT_PATH, num_classes=1000, device=device)
    
    # Get validation dataloader
    print(f"\nLoading validation data from {VAL_DATA_PATH}")
    val_loader, val_dataset = get_val_dataloader(batch_size=64, num_workers=4)
    print(f"Validation dataset size: {len(val_dataset)} images")
    
    # Evaluate model
    metrics = evaluate_model(model, val_loader, device=device)
    
    # Get misclassified images (single batch for speed)
    print("\nFinding misclassified examples...")
    single_batch_loader, _ = get_val_dataloader(batch_size=1, num_workers=4)
    misclassified_data = get_misclassified_images(
        model, 
        single_batch_loader, 
        classes, 
        device=device, 
        max_images=30,
        use_gradcam=True  # Set to True to generate GradCAM visualizations (slower)
    )
    
    # Plot misclassified images
    if len(misclassified_data['images']) > 0:
        plot_misclassified_images(
            misclassified_data, 
            classes, 
            save_path='misclassified_report.png',
            use_gradcam=True
        )
        plot_misclassified_images(
            misclassified_data, 
            classes, 
            save_path='misclassified_report_no_gradcam.png',
            use_gradcam=False
        )
        
        # Save individual images
        save_individual_misclassified_images(misclassified_data, classes)
    
    # Generate report
    print("\n" + "="*70)
    print("INFERENCE REPORT COMPLETE")
    print("="*70)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {device}")
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
    print(f"Misclassified examples saved: {len(misclassified_data['images'])}")
    print("="*70)

if __name__ == "__main__":
    main()