import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
import argparse
import sys
import os
import yaml
from functools import partial
from importlib import import_module
from utils.util import set_all_seeds
from utils.metrics import Metrics

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_class_from_string(class_path):
    module_path, class_name = class_path.rsplit('.', 1)
    module = import_module(module_path)
    return getattr(module, class_name)

def get_function_from_string(func_path):
    module_path, func_name = func_path.rsplit('.', 1)
    module = import_module(module_path)
    return getattr(module, func_name)

def instantiate_model(config, num_classes=None, model_path=None):
    model_class = get_class_from_string(config['model_class'])
    model_configs = config['model_configs'].copy()
    
    if num_classes is not None:
        model_configs['num_classes'] = num_classes
    
    if config['model_name'] == 'VideoMAE-v2':
        model_configs['norm_layer'] = partial(nn.LayerNorm, eps=1e-6)
    
    model = model_class(**model_configs)
    
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model_name_lower = config['model_name'].lower()
        
        if 'model_state_dict' in checkpoint:
            if 'videomae' in model_name_lower:
                checkpoint['model_state_dict'].pop('head.weight', None)
                checkpoint['model_state_dict'].pop('head.bias', None)
            elif 'x3d' in model_name_lower:
                checkpoint['model_state_dict'].pop('cls_head.fc2.weight', None)
                checkpoint['model_state_dict'].pop('cls_head.fc2.bias', None)
            elif 'i3d' in model_name_lower:
                checkpoint['model_state_dict'].pop('logits.conv3d.weight', None)
                checkpoint['model_state_dict'].pop('logits.conv3d.bias', None)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            if 'videomae' in model_name_lower:
                checkpoint.pop('head.weight', None)
                checkpoint.pop('head.bias', None)
            elif 'x3d' in model_name_lower:
                checkpoint.pop('cls_head.fc2.weight', None)
                checkpoint.pop('cls_head.fc2.bias', None)
            elif 'i3d' in model_name_lower:
                checkpoint.pop('logits.conv3d.weight', None)
                checkpoint.pop('logits.conv3d.bias', None)
            model.load_state_dict(checkpoint, strict=False)
        
        print(f"Successfully loaded model weights from {model_path}")
    
    return model

def create_transformations(config, is_train=True):
    dataset_config = config['train_dataset_configs'] if is_train else config['validation_dataset_configs']
    transform_func = get_function_from_string(dataset_config['transform'])
    transform_configs = dataset_config['transformation_configs']
    
    return transform_func(**transform_configs)

def train_one_epoch(model, train_loader, optimizer, criterion, device, accumulation_steps, use_wandb):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    accumulated_loss = 0.
    
    for i, batch in enumerate(progress_bar):
        inputs, labels = batch['pixel_values'], batch['labels']
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps
        accumulated_loss += loss.item()
        running_loss += loss.item()
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            
            progress_bar.set_postfix(loss=accumulated_loss)
            
            optimizer.step()
            optimizer.zero_grad()
            
            if use_wandb:
                wandb.log({"batch_loss": accumulated_loss, "lr": optimizer.param_groups[0]['lr']})
            
            accumulated_loss = 0.
    
    return running_loss * accumulation_steps / len(train_loader)


def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    metric = Metrics(num_classes = 129, k = 3)

    progress_bar = tqdm(val_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            inputs, labels = batch['pixel_values'], batch['labels']
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            metric.update(outputs, labels)
            
    val_loss = running_loss / len(val_loader)
    perf = metric.compute()
    
    return val_loss, perf


def train_model(model: torch.nn.Module, 
                train_loader: torch.utils.data.DataLoader, 
                val_loader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, 
                epochs: int, 
                device: torch.device, 
                accumulation_steps: int,  
                save_ckpt_every: int,
                save_ckpt_dir: str,
                scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                use_wandb: bool = False):
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    model = model.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, accumulation_steps, use_wandb)
        
        print(f"Training loss: {train_loss:.4f}")

        if val_loader is not None:
            val_loss, perf = validate_model(model, val_loader, criterion, device)
            print(f"Validation loss: {val_loss:.4f}")
            print(f"Validation accuracy: {perf['mean_accuracy']:.4f}")
            print(f"Validation F1: {perf['mean_f1']:.4f}")
            print(f"Validation Precision: {perf['mean_precision']:.4f}")
            print(f"Validation Recall: {perf['mean_recall']:.4f}")
            print(f"Validation Top-K Accuracy: {perf['top_k_accuracy']:.4f}")
            
            if use_wandb:
                wandb.log({
                           "epoch": epoch + 1,
                           "val_loss": val_loss, 
                           "train_loss": train_loss, 
                           "val_accuracy": perf['mean_accuracy'],
                           "val_f1": perf['mean_f1'],
                           "val_precision": perf['mean_precision'],
                           "val_recall": perf['mean_recall'],
                           "val_top_k_accuracy": perf['top_k_accuracy'],
                        }) 
        
        elif val_loader is None and use_wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss}) 
    
        if (epoch + 1) % save_ckpt_every == 0:     
            torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                }, f'{save_ckpt_dir}/model_epoch_{epoch + 1}.pth')

        if scheduler:
            scheduler.step()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config YAML file')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of classes (overrides config)')
    parser.add_argument('--model_path', type=str,
                        help='Path to pretrained model weights (overrides config)')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers for data loading')
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Train data directory (overrides config)')
    parser.add_argument('--validation_data_path', type=str,
                        help='Validation data directory (overrides config)')
    parser.add_argument('--test_data_path', type=str,
                        help='Test data directory')
    parser.add_argument('--warmup_steps', type=float, default=0.0,
                        help='Percentage of total training steps for warmup')
    parser.add_argument('--save_ckpt_every', type=int, default=1,
                        help='Number of epochs between checkpoints')
    parser.add_argument('--save_ckpt_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Number of accumulation steps')
    parser.add_argument('--scheduler', type=str, default=None,
                        help='Type of learning rate scheduler')
    parser.add_argument('--class_balance', type=str2bool, nargs='?', const=True, default=False,
                        help='Handle class imbalance')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training')
    parser.add_argument('--use_wandb', type=str2bool, nargs='?', const=True, default=False,
                        help='Use wandb for logging')
    args = parser.parse_args()

    config = load_config(args.config)
    
    model = instantiate_model(config, num_classes=args.num_classes, model_path=args.model_path)
    
    train_transformations = create_transformations(config, is_train=True)
    val_transformations = create_transformations(config, is_train=False)

    if args.use_wandb:
        import wandb
        from dotenv import load_dotenv
        
        load_dotenv()
        wandb_api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=wandb_api_key)
        wandb.init(project='classification_model', config={"monitor_gpus": True})
    
    if not os.path.exists(args.save_ckpt_dir):
        os.makedirs(args.save_ckpt_dir)

    set_all_seeds()
    
    from utils.prepare_dataset import VideoDataset
    from utils.util import collate_fn
    
    train_dataset = VideoDataset(root_dir=args.train_data_path, transform=train_transformations)
    val_dataset = VideoDataset(root_dir=args.validation_data_path, transform=val_transformations)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, sampler=train_dataset.get_sampler())
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    if args.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    else:
        scheduler = None
        
    train_model(model=model,
                train_loader=train_dataloader,
                val_loader=val_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=args.num_epochs,
                device=args.device,
                accumulation_steps=args.accumulation_steps,
                save_ckpt_every=args.save_ckpt_every,
                save_ckpt_dir=args.save_ckpt_dir,
                use_wandb=args.use_wandb
                )