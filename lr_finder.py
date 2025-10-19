#------------------------------------AMP START------------------------------------
from torch_lr_finder import LRFinder
import torch
import torchvision
import torchvision.transforms as transforms
from model import ResNet50Wrapper
import torch.nn as nn
import torch.optim as optim
from utils import Config
import fire
import matplotlib.pyplot as plt
from datetime import datetime
import os
from torch.cuda.amp import autocast, GradScaler

# Optional but helpful for fragmentation avoidance
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def find_lr(start_lr=1e-6, end_lr=1, num_iter=500, output_dir="lr_finder_plots"):
    config = Config()
    print(f"🔍 Finding LR with Start={start_lr}, End={end_lr}, Num_iter={num_iter}")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"🧠 Using device: {device}")

    # --- Dataset setup ---
    train_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(
            config.IMG_W,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=config.train_folder_name,
        transform=train_transformation,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,  # Keep your full batch size
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
    )

    # --- Model setup ---
    model = ResNet50Wrapper(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=start_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    # --- Mixed precision setup ---
    scaler =  torch.amp.GradScaler("cuda")
    lr_finder = LRFinder(model, optimizer, criterion, device=device)

    # Override LR Finder’s train batch to use AMP
    def amp_train_batch(self, inputs, targets=None, **kwargs):
    # If inputs is a DataLoader iterator, extract the batch
        if not torch.is_tensor(inputs):
            try:
                inputs, targets = next(inputs)
            except TypeError:
                raise ValueError("Invalid batch input received by amp_train_batch")

        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.item()


    lr_finder._train_batch = amp_train_batch.__get__(lr_finder, LRFinder)

    # --- Clear memory and run test ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("🚀 Starting LR range test...")
    lr_finder.range_test(
        train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode="exp",
    )

    # --- Save plot ---
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lr_finder_{timestamp}_start{start_lr}_end{end_lr}_iter{num_iter}.png"
    filepath = os.path.join(output_dir, filename)

    fig, ax = plt.subplots()
    lr_finder.plot(ax=ax)
    plt.title(f"Learning Rate Finder (iter={num_iter})")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ LR Finder plot saved to: {filepath}")
    lr_finder.reset()


if __name__ == "__main__":
    fire.Fire(find_lr)

#------------------------------------AMP END------------------------------------
# from torch_lr_finder import LRFinder
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from model import ResNet50Wrapper
# import torch.nn as nn
# import torch.optim as optim
# from utils import Config, AMPConfig
# import fire
# import matplotlib.pyplot as plt
# from datetime import datetime
# import os
# from torch.cuda.amp import autocast, GradScaler
# # import torch
# # torch.set_float32_matmul_precision('high')

# def find_lr(start_lr=1e-7, end_lr=10, num_iter=100, output_dir='lr_finder_plots'):
#     config = Config()
#     print(f"Find LR with params: Start_lr: {start_lr}, End_lr: {end_lr}, Num_iter: {num_iter}")
#     device = (
#         "cuda"
#         if torch.cuda.is_available()
#         else "mps"
#         if torch.backends.mps.is_available()
#         else "cpu"
#     )

#     print(f"Using {device} device")
    
#     training_folder_name = config.train_folder_name
#     train_transformation = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.RandomResizedCrop(config.IMG_W, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
#         transforms.RandomHorizontalFlip(0.5),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     train_dataset = torchvision.datasets.ImageFolder(
#         root=training_folder_name,
#         transform=train_transformation
#     )
    
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=config.batch_size,
#         shuffle=True,
#         num_workers=config.workers,
#         pin_memory=True
#     )

#     model = ResNet50Wrapper(num_classes=len(train_dataset.classes)).to(device)
#     #model.use_gradient_checkpointing()  # Enable gradient checkpointing
#     # model = torch.compile(model, mode="reduce-overhead")  # Enable torch.compile
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(
#         model.parameters(), 
#         lr=start_lr, 
#         momentum=config.momentum, 
#         weight_decay=config.weight_decay
#     )

#     scaler = GradScaler()
#     lr_finder = LRFinder(
#         model, 
#         optimizer, 
#         criterion, 
#         device=device
#     )
#     def amp_train_batch(self, inputs, targets):
#         optimizer.zero_grad()
#         with autocast():
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         return loss.item()
#     lr_finder._train_batch = amp_train_batch.__get__(lr_finder, LRFinder)
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()
#     lr_finder.range_test(train_loader, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter, step_mode="exp")
    
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Generate filename with timestamp and parameters
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     filename = f'lr_finder_{timestamp}_start{start_lr}_end{end_lr}_iter{num_iter}.png'
#     filepath = os.path.join(output_dir, filename)
    
#     # Plot and save
#     fig, ax = plt.subplots()
#     lr_finder.plot(ax=ax)
#     plt.title(f'Learning Rate Finder (iter: {num_iter})')
#     plt.savefig(filepath, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Plot saved to: {filepath}")
#     lr_finder.reset()

# if __name__ == "__main__":
#     fire.Fire(find_lr)