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
from utils import CSVLogger, upload_file_to_s3

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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import sys




def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, dataloader, model, loss_fn, optimizer, scheduler, epoch, writer, scaler, csv_logger):
    logger.info(f"[Train Stage] Epoch {epoch+1} Stage : Training STARTS")
    size = len(dataloader.dataset)
    model.train()
    start0 = time.time()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    if rank == 0:
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
    else:
        progress_bar = enumerate(dataloader)
    optimizer.zero_grad(set_to_none=True)
    for batch, (X, y) in progress_bar:
        X, y = X.cuda(rank), y.cuda(rank)

        with torch.amp.autocast("cuda"):
            pred = model(X)
            loss = loss_fn(pred, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #optimizer.zero_grad()

        scheduler.step()

        running_loss += loss.item()
        total += y.size(0)

        _, predicted = torch.max(pred.data, 1)
        correct += (predicted == y).sum().item()

        _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
        correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()

        if rank == 0 and batch % 100 == 0:
            current_loss = loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            current_acc = 100 * correct / total
            current_acc5 = 100 * correct_top5 / total

            if isinstance(progress_bar, tqdm):
                progress_bar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "acc": f"{current_acc:.3f}%",
                    "acc5": f"{current_acc5:.3f}%",
                    "lr": f"{current_lr:.7f}"
                })

    # Gather metrics from all processes
    world_size = dist.get_world_size()
    running_loss_tensor = torch.tensor([running_loss]).cuda(rank)
    correct_tensor = torch.tensor([correct]).cuda(rank)
    correct_top5_tensor = torch.tensor([correct_top5]).cuda(rank)
    total_tensor = torch.tensor([total]).cuda(rank)

    dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_top5_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    epoch_time = time.time() - start0
    avg_loss = running_loss_tensor.item() / (len(dataloader) * world_size)
    accuracy = 100 * correct_tensor.item() / total_tensor.item()
    accuracy_top5 = 100 * correct_top5_tensor.item() / total_tensor.item()

    if rank == 0:
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
        logger.info(f"[Train Stage] Epoch {epoch+1} - Loss: {avg_loss:.4f}, Acc: {accuracy:.3f}%, Top-5 Acc: {accuracy_top5:.3f}%, Time: {epoch_time:.2f}s LR: {optimizer.param_groups[0]['lr']:.7f}")
    return metrics if rank == 0 else None

def test(rank, dataloader, model, loss_fn, epoch, writer, train_dataloader, csv_logger, calc_acc5=True):
    logger.info(f"[Test Stage] Epoch {epoch+1} Stage : Testing STARTS")
    model.eval()
    test_loss = 0
    correct = 0
    correct_top5 = 0
    total = 0

    if rank == 0:
        progress_bar = tqdm(dataloader, desc=f"Testing Epoch {epoch+1}")
    else:
        progress_bar = dataloader

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            for X, y in progress_bar:
                X, y = X.cuda(rank), y.cuda(rank)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                total += y.size(0)
                
                _, predicted = torch.max(pred.data, 1)
                correct += (predicted == y).sum().item()
                
                if calc_acc5:
                    _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
                    correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()

    # Gather metrics from all processes
    world_size = dist.get_world_size()
    test_loss_tensor = torch.tensor([test_loss]).cuda(rank)
    correct_tensor = torch.tensor([correct]).cuda(rank)
    correct_top5_tensor = torch.tensor([correct_top5]).cuda(rank)
    total_tensor = torch.tensor([total]).cuda(rank)

    dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_top5_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = test_loss_tensor.item() / len(dataloader)
        accuracy = 100 * correct_tensor.item() / total_tensor.item()
        accuracy_top5 = 100 * correct_top5_tensor.item() / total_tensor.item() if calc_acc5 else None

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
    return None

def main_worker(rank, world_size, config, args):
    try:
        setup(rank, world_size)
        
        if rank == 0:
            os.makedirs(os.path.join("checkpoints", config.name), exist_ok=True)
            os.makedirs(os.path.join("logs", config.name, "app_logs"), exist_ok=True)
            os.makedirs(os.path.join("logs", config.name, "csv_logger"), exist_ok=True)

        # Wait for rank 0 to create directories
        dist.barrier()

        # Set up logging only for rank 0
        if rank == 0:
            log_filename = f"training_apps.log"
            log_filepath = os.path.join("logs", config.name, "app_logs", log_filename)
            
            file_handler = logging.FileHandler(log_filepath)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(formatter)
            
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            logger.addHandler(file_handler)
        
        # Create metric logger
        log_dir = os.path.join("logs", config.name, 'csv_logger')
        csv_logger = CSVLogger(log_dir) if rank == 0 else None

        # Data loading code
        train_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = torchvision.datasets.ImageFolder(
            root=config.train_folder_name,
            transform=train_transformation
        )
        
        train_sampler = DistributedSampler(train_dataset)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=config.workers,
            pin_memory=True,
            persistent_workers=True
        )

        val_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=256, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_dataset = torchvision.datasets.ImageFolder(
            root=config.val_folder_name,
            transform=val_transformation
        )
        
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            sampler=val_sampler,
            num_workers=config.workers,
            pin_memory=True
        )

        torch.cuda.set_device(rank)
        
        num_classes = len(train_dataset.classes)
        model = ResNet50Wrapper(num_classes=num_classes).cuda(rank)
        model = DDP(model, device_ids=[rank])

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),
                                  lr=config.max_lr/config.div_factor,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

        scaler = torch.amp.GradScaler("cuda")
        
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

        start_epoch = 0
        if args.resume and os.path.exists(args.checkpoint_path):
            if rank == 0:
                logger.info("Resuming training from checkpoint")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(args.checkpoint_path, map_location=map_location)
            model.module.load_state_dict(checkpoint["model"])
            start_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])

        writer = None
        test(rank, val_loader, model, loss_fn, epoch=0, writer=writer, 
             train_dataloader=train_loader, csv_logger=csv_logger, calc_acc5=True)

        if rank == 0:
            logger.info(f"Starting training from epoch {start_epoch}")

        for epoch in range(start_epoch, config.epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            
            if rank == 0:
                logger.info(f"Current Epoch {epoch}")
            
            train_metrics = train(rank, train_loader, model, loss_fn, optimizer, scheduler, 
                                epoch=epoch, writer=writer, scaler=scaler, csv_logger=csv_logger)
            
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "config": config
                }
                
                # Save checkpoint and upload to S3
                checkpoint_name = f"checkpoint-epoch-{epoch}-train_acc-{train_metrics['accuracy']:.2f}-test_acc-{test_metrics['accuracy']:.2f}-train_acc5-{train_metrics['accuracy_top5']:.2f}-test_acc5-{test_metrics['accuracy_top5']:.2f}.pth"
                checkpoint_path = os.path.join("checkpoints", config.name, checkpoint_name)
                
                # Create checkpoint directory if it doesn't exist
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                
                # Save checkpoint locally
                logger.info(f"Saving checkpoint to {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)

                # Compress checkpoint before uploading
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

                # Upload logs
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

                # Upload app logs
                try:
                    log_filepath = os.path.join("logs", config.name, "app_logs", "training_apps.log")
                    s3_uri = upload_file_to_s3(
                        log_filepath,
                        bucket_name='resnet-1000',
                        s3_prefix='log_handler/epoch_'+str(epoch)
                    )
                    logger.info(f"Log file upload completed: {s3_uri}")
                except Exception as e:
                    logger.error(f"Failed to upload log file to S3: {str(e)}")
                    raise

                logger.info(f"All uploads completed successfully for epoch {epoch}")
                
            test_metrics = test(rank, val_loader, model, loss_fn, epoch + 1, writer,
                              train_dataloader=train_loader, csv_logger=csv_logger, calc_acc5=True)

    except Exception as e:
        print(f"Error in rank {rank}: {str(e)}")
        raise e
    finally:
        cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ResNet50 Training')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to checkpoint file for resuming training')
    parser.add_argument('--resume', type=bool, default=False,
                       help='Resume training from checkpoint')
    args = parser.parse_args()
    
    config = Config()
    
    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")
    try:
        mp.spawn(main_worker,
                args=(world_size, config, args),
                nprocs=world_size,
                join=True)
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        sys.exit(1)

        




