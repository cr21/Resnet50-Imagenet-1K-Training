import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import time
from math import sqrt
from utils import JsonLogger, CSVLogger, upload_file_to_s3

from model import ResNet50Wrapper
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from utils import Config
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import logging
from datetime import datetime




def train(dataloader, model, loss_fn, optimizer, scheduler, epoch, writer, scaler, csv_logger):
    logger.info(f"[Train Stage] Epoch {epoch+1} Stage : Training STARTS")
    size = len(dataloader.dataset)
    model.train()
    start0 = time.time()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")

    for batch, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)

        # amp mixed precision training
        with autocast():
            pred = model(X)
            loss = loss_fn(pred, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Step the scheduler here, after each batch for onecycle lr scheduler
        scheduler.step()

        running_loss += loss.item()
        total += y.size(0)

        # Calculate accuracy    for top 1 accuracy
        _, predicted = torch.max(pred.data, 1)
        correct += (predicted == y).sum().item()

        # Calculate top-5 accuracy
        _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
        correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()

        batch_size = len(X)
        step = epoch * size + (batch + 1) * batch_size

        if batch % 100 == 0:
            current_loss = loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            current_acc = 100 * correct / total
            current_acc5 = 100 * correct_top5 / total

            progress_bar.set_postfix({
                "train_loss": f"{current_loss:.4f}",
                "train_acc": f"{current_acc:.3f}%",
                "train_acc5": f"{current_acc5:.3f}%",
                "lr": f"{current_lr:.7f}"
            })

    epoch_time = time.time() - start0
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    accuracy_top5 = 100 * correct_top5 / total

    metrics = {
        'stage': 'train',
        'epoch': epoch + 1,
        'loss': avg_loss,
        'accuracy': accuracy,
        'accuracy_top5': accuracy_top5,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'epoch_time': epoch_time
    }
    csv_logger.log_metrics(metrics)

    logger.info(f"[Train Stage] Epoch {epoch+1} Stage : Training - Loss: {avg_loss:.4f}, Acc: {accuracy:.3f}%, Top-5 Acc: {accuracy_top5:.3f}%, Time: {epoch_time:.2f}s LR: {optimizer.param_groups[0]['lr']:.7f}")
    return metrics

def test(dataloader, model, loss_fn, epoch, writer, train_dataloader, csv_logger, calc_acc5=True):
    logger.info(f"[Test Stage] Epoch {epoch+1} Stage : Testing STARTS")
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    correct_top5 = 0

    progress_bar = tqdm(dataloader, desc=f"Testing Epoch {epoch+1}")

    with torch.no_grad():
        with autocast():
            for X, y in progress_bar:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                
                _, predicted = torch.max(pred.data, 1)
                correct += (predicted == y).sum().item()
                
                if calc_acc5:
                    _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
                    correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()
                
                current_acc = 100 * correct / size
                current_acc5 = 100 * correct_top5 / size if calc_acc5 else 0
                current_loss = test_loss / (progress_bar.n + 1)
                
                progress_bar.set_postfix({
                    "test_loss": f"{current_loss:.4f}",
                    "test_acc": f"{current_acc:.2f}%",
                    "test_acc5": f"{current_acc5:.2f}%"
                })

    test_loss /= num_batches
    accuracy = 100 * correct / size
    accuracy_top5 = 100 * correct_top5 / size if calc_acc5 else None

    metrics = {
        'stage': 'test',
        'epoch': epoch + 1,
        'loss': test_loss,
        'accuracy': accuracy,
        'accuracy_top5': accuracy_top5
    }
    csv_logger.log_metrics(metrics)

    logger.info(f"[Test Stage] Epoch {epoch+1} - Loss: {test_loss:.4f}, Acc: {accuracy:.2f}%, Top-5 Acc: {accuracy_top5:.2f}%")
    return metrics

if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description='ResNet50 Training')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint file for resuming training')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume training from checkpoint')
    args = parser.parse_args()

    config = Config()
    
    ## SET UP LOGGING
    # Update logging configuration
    log_filename = f"training_apps.log"
    log_filepath = os.path.join("logs", config.name, "app_logs", log_filename)
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

    # Configure file handler with timestamp format
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)



    # Create metric logger
    log_dir = os.path.join("logs", config.name,'csv_logger')
    csv_logger = CSVLogger(log_dir)


    training_folder_name = config.train_folder_name
    val_folder_name = config.val_folder_name

    train_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=training_folder_name,
        transform=train_transformation
    )
    train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.workers,
        pin_memory=True,
        prefetch_factor=1 # TODO: check if this is needed
    )

    val_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=256, antialias=True),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=val_folder_name,
        transform=val_transformation
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=512, # large batch size for validation
        num_workers=config.workers,
        shuffle=False,
        pin_memory=True
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using {device} device")

    resume_training = args.resume # make this False to start from scratch
    logger.info(f"Resume training: {resume_training}")
    num_classes = len(train_dataset.classes)
    model = ResNet50Wrapper(num_classes=num_classes)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss() # TODO : can experiment with smoothing loss
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=config.max_lr/config.div_factor, # as we are using onecycle lr scheduler inital lr is set to max_lr/div_factor
                               momentum=config.momentum,
                               weight_decay=config.weight_decay)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()
    steps_per_epoch = len(train_loader)
    total_steps = config.epochs * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        total_steps=total_steps,
        pct_start=config.pct_start,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor
    )
    # One Cycle LR Scheduler
    # Rate                     peak_lr (max_lr)
    # ^                        ╭─╮
    # │                       /   \
    # │                     /      \
    # │                   /         \
    # │                 /            \
    # │               /               \
    # │             /                  \
    # │           /                     \
    # │         /                        \
    # │       /                           \
    # initial_lr                              \
    # │     (max_lr/div_factor)            \
    # │                                     \  final_lr
    # │                                      \ (max_lr/(div_factor*final_div_factor))
    # └──────────────────────────────────────┴─────▶
    # 0        pct_start                           100%
    #             (e.g., 30%)                     Training Progress

    # TODO : add early stopping

    start_epoch = 0
    checkpoint_path = args.checkpoint_path or os.path.join("checkpoints", config.name, f"checkpoint.pth")
    
    if resume_training and os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        print(config)
        print(checkpoint["config"])

    writer = None # TODO : add tensorboard or any other metric logger
    test(val_loader, model, loss_fn, epoch=0, writer=writer, train_dataloader=train_loader, 
         csv_logger=csv_logger, calc_acc5=True)
    
    logger.info(f"Starting training from epoch { start_epoch}")
    for epoch in range(start_epoch, config.epochs):
        logger.info(f"Current Epoch {epoch}")
        train_metrics = train(train_loader, model, loss_fn, optimizer, scheduler, epoch=epoch, writer=writer, 
              scaler=scaler, csv_logger=csv_logger)
        
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "config": config
        }
        #torch.save(checkpoint, os.path.join("checkpoints", config.name, f"model_{epoch}.pth"))
        #torch.save(checkpoint, os.path.join("checkpoints", config.name, f"checkpoint.pth"))
       
        logger.info(f"Current Epoch {epoch} Training completed")
        test_metrics = test(val_loader, model, loss_fn, epoch + 1, writer, train_dataloader=train_loader,
             csv_logger=csv_logger, calc_acc5=True)
        logger.info(f"Current Epoch {epoch} Testing completed")

        # save the model checkpoint and logger to s3 
        checkpoint_name = f"checkpoint-epoch-{epoch}-train_acc-{train_metrics['accuracy']:.2f}-test_acc-{test_metrics['accuracy']:.2f}-train_acc5-{train_metrics['accuracy_top5']:.2f}-test_acc5-cle{test_metrics['accuracy_top5']:.2f}.pth"
        checkpoint_path = os.path.join("checkpoints", config.name, checkpoint_name)
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Save checkpoint locally
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)

        # Compress checkpoint before uploading (optional)
        import gzip
        compressed_path = checkpoint_path + '.gz'
        with open(checkpoint_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Upload compressed checkpoint with progress tracking
        logger.info(f"Starting upload of checkpoint ({os.path.getsize(compressed_path)/1024/1024:.2f} MB)")
        try:
            s3_uri = upload_file_to_s3(
                compressed_path,
                bucket_name='resnet-1000',
                s3_prefix='imagenet1K_epoch_/epoch_'+str(epoch)
            )
            logger.info(f"Model checkpoint upload completed:")
        except Exception as e:
            logger.error(f"Failed to upload checkpoint to S3: {str(e)}")
            raise

        # Upload logs (these are typically much smaller)
        for log_name, prefix in [
            ('training_log.csv', 'imagenet1K-csv-train-logs'),
            ('test_log.csv', 'imagenet1K-csv-test-logs')
        ]:
            try:
                log_path = os.path.join("logs", config.name, 'csv_logger', log_name)
                s3_uri = upload_file_to_s3(
                    log_path,
                    bucket_name='resnet-1000',
                    s3_prefix=prefix+'/epoch_'+str(epoch)
                )
                logger.info(f"{log_name} upload completed: ")
            except Exception as e:
                logger.error(f"Failed to upload {log_name} to S3: {str(e)}")
                raise

        # Upload logs to S3
        try:
            s3_uri = upload_file_to_s3(
                log_filepath,
                bucket_name='resnet-1000',
                s3_prefix='log_handler/epoch_'+str(epoch)
            )
            logger.info(f"Log file upload completed: {s3_uri}")
        except Exception as e:
            logger.error(f"Failed to upload log file to S3: {str(e)}")
            raise

        logger.info("All uploads completed successfully for epoch ", epoch)

        




