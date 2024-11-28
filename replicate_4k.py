import os
import re
from itertools import chain
import torch
import numpy as np
from tqdm import tqdm
import wandb
import argparse
import psutil
import gc
import emd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import itertools
import datetime

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        """
        Initialize the dataset with input features X and labels Y.
        Args:
            X (torch.Tensor): The input features tensor of shape (N, 4, 1250).
            Y (torch.Tensor): The labels tensor of shape (N, 2).
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        Args:
            idx (int): Index of the data item.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple (X[idx], Y[idx]).
        """
        return self.X[idx], self.Y[idx]


def collate_fn(batch):
    """
    Custom collate function to filter data in a batch.
    Args:
        batch (list): List of (X, Y) tuples from the dataset.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Filtered batch (X_batch, Y_batch).
    """
    X_batch, Y_batch = zip(*batch)
    X_batch = torch.stack(X_batch)
    Y_batch = torch.stack(Y_batch)

    # Apply the mask to filter the data
    mask = torch.any(X_batch[:, 3, :] != 0, dim=1)
    return X_batch[mask], Y_batch[mask]


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout=0.1):
        super().__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_embedding = nn.Embedding(1250, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.mlm_head = nn.Linear(hidden_size, input_size)
        self.pred_head1 = nn.Linear(hidden_size, 128)
        self.pred_head2 = nn.Linear(128, 2)

    def forward(self, x, mask=None):
        x = self.embedding(x)

        positions = torch.arange(x.size(1), dtype=torch.long, device=x.device)
        pos_embeddings = self.pos_embedding(positions).unsqueeze(0)
        x = x + pos_embeddings

        if mask is not None:
            mask = mask.transpose(0, 1)

        x = self.encoder(x, src_key_padding_mask=mask)

        mlm_logits = self.mlm_head(x)
        prediction = self.pred_head2(self.pred_head1(x[:, 0]))
        return mlm_logits, prediction

def train_mlm(config, train_loader, test_loader, save_path=None):
    """
    Train MLM model with comprehensive WandB logging and save capabilities.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract hyperparameters from config
    input_size = config['input_size']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    dropout = config['dropout']
    mask_ratio = config['mask_ratio']
    lr = config['learning_rate']
    patience = config['patience']
    
    model = TransformerEncoder(input_size, hidden_size, num_layers, num_heads, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience)
    
    # Initialize tracking variables
    best_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(100):
        model.train()
        total_mlm_loss = 0.0
        batch_losses = []
        
        # Training phase
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} - Training", leave=False)
        for batch_idx, (X_batch, *_) in enumerate(progress_bar):
            X_batch = X_batch.float().permute(0, 2, 1).to(device)
            
            # Prepare mask for MLM
            mask = torch.rand(X_batch.size()[:2], device=device) < mask_ratio
            original_data = X_batch.clone()
            X_batch[mask] = 0.0
            
            # Forward pass
            mlm_logits, *_ = model(X_batch)
            mlm_loss = nn.MSELoss()(mlm_logits[mask], original_data[mask])
            
            # Backpropagation
            optimizer.zero_grad()
            mlm_loss.backward()
            optimizer.step()
            
            # Track batch-level metrics
            batch_loss = mlm_loss.item()
            total_mlm_loss += batch_loss
            batch_losses.append(batch_loss)
            
            # Log batch-level metrics
            wandb.log({
                'batch': epoch * len(train_loader) + batch_idx,
                'batch_loss': batch_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Update progress bar
            progress_bar.set_postfix({'batch_loss': f'{batch_loss:.4f}'})
        
        avg_train_loss = total_mlm_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_batch_losses = []
        
        with torch.no_grad():
            for X_batch, *_ in tqdm(test_loader, desc=f"Epoch {epoch} - Validation", leave=False):
                X_batch = X_batch.float().permute(0, 2, 1).to(device)
                mask = torch.rand(X_batch.size()[:2], device=device) < mask_ratio
                original_data = X_batch.clone()
                X_batch[mask] = 0.0
                mlm_logits, *_ = model(X_batch)
                val_loss = nn.MSELoss()(mlm_logits[mask], original_data[mask])
                total_val_loss += val_loss.item()
                val_batch_losses.append(val_loss.item())
        
        avg_val_loss = total_val_loss / len(test_loader)
        
        # Learning rate scheduling
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track if learning rate changed
        if current_lr != prev_lr:
            wandb.log({'lr_decreased': True, 'epoch': epoch})
        
        # Log epoch-level metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_loss_std': np.std(batch_losses),
            'val_loss_std': np.std(val_batch_losses),
            'learning_rate': current_lr,
        })
        
        # Model checkpointing with best weights saving
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_without_improvement = 0
            
            # Store the best model state dictionary
            best_model_state = model.state_dict().copy()
            
            if save_path:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'config': config
                }
                torch.save(checkpoint, save_path)
                wandb.log({'saved_checkpoint': True, 'epoch': epoch, 'checkpoint_path': save_path})
        else:
            epochs_without_improvement += 1
            
        # Early stopping check
        if epochs_without_improvement >= patience * 2:  # 2x scheduler patience for early stopping
            print(f"Early stopping triggered after {epoch} epochs")
            wandb.log({'early_stopping': True, 'epoch': epoch})
            break
    
    # Optional: Return the best model state for further use
    return best_loss, best_model_state


save_dir = './model_checkpoints'
os.makedirs(save_dir, exist_ok=True)
def grid_search(train_loader, test_loader, save_dir='./model_checkpoints'):
    param_grid = {
        'hidden_size': [32, 64, 128],
        'num_layers': [3],
        'num_heads': [8, 4],
        'dropout': [0.3],
        'mask_ratio': [0.3],
        'learning_rate': [1e-3],
        'patience': [5],
        'input_size': [4]
    }
    
    best_config = None
    best_val_loss = float('inf')
    
    # Create a unique group ID for this grid search
    grid_search_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate total number of combinations
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    
    for run_idx, params in enumerate(tqdm(list(dict(zip(param_grid.keys(), values)) 
                                    for values in itertools.product(*param_grid.values())))):
        
        # Initialize WandB with group tracking
        wandb.init(
            project="ppg-grid-search",
            config=params,
            group=f"grid_search_{grid_search_id}",
            name=f"run_{run_idx}",
            reinit=True,
            tags=[f"{k}_{v}" for k, v in params.items()]
        )
        
        # Log grid search progress
        wandb.log({
            'grid_search_progress': run_idx / total_combinations * 100,
            'runs_completed': run_idx,
            'runs_remaining': total_combinations - run_idx
        })

        checkpoint_path = os.path.join(save_dir, f'checkpoint_run_{run_idx}_{grid_search_id}.pt')
        
        # Train model and get validation loss
        #val_loss, model_state = train_mlm(params, train_loader, test_loader)
        val_loss, model_state = train_mlm(params, train_loader, test_loader, save_path=checkpoint_path)

        # Log final metrics
        wandb.log({
            'final_val_loss': val_loss,
            'run_completed': True
        })
        
        # Update best configuration
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = params
            best_model_state = model_state
            wandb.log({
                'new_best_model': True,
                'best_val_loss': best_val_loss
            })
        
        wandb.finish()
    
    # Log final results
    wandb.init(
        project="ppg-grid-search",
        name="grid_search_summary",
        group=f"grid_search_{grid_search_id}",
        reinit=True
    )
    
    wandb.log({
        'best_config': best_config,
        'best_val_loss': best_val_loss,
        'total_runs': total_combinations
    })
    
    # Create a summary table of all configurations
    wandb.Table(
        columns=['run_id'] + list(param_grid.keys()) + ['val_loss'],
        data=[[i] + list(params.values()) + [loss] for i, (params, loss) in enumerate(results)]
    )
    
    wandb.finish()
    
    print(f"Best config: {best_config}, Best validation loss: {best_val_loss}")
    return best_config, best_val_loss

def train_prediction_across_checkpoints(train_loader, test_loader, checkpoint_dir='./model_checkpoints'):
    """
    Train prediction models for all MLM checkpoints in the specified directory
    
    Args:
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        checkpoint_dir (str): Directory containing MLM model checkpoints
    
    Returns:
        dict: Dictionary of trained models and their performance
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Calculate target mean and standard deviation from training labels
    Y_train_tensor = torch.stack([y for _, y in train_loader.dataset])
    target_mean = Y_train_tensor.mean(dim=0)
    target_std = Y_train_tensor.std(dim=0)
    
    # Get all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    results = {}
    
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        wandb.init(
            project="prediction-training",
            name=f"prediction_{os.path.splitext(checkpoint_file)[0]}",
            reinit=True
        )
        
        # Load checkpoint to get configuration
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        
        # Initialize model with saved configuration
        model = TransformerEncoder(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(device)
        
        # Load pre-trained model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Prepare optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        # Training loop similar to previous implementation
        best_loss = float('inf')
        for epoch in tqdm(range(100)):  # Adjust number of epochs as needed
            model.train()
            total_pred_loss = 0.0
            all_predictions = []
            all_true_labels = []
            
            for X_batch, Y_batch in tqdm(train_loader):
                X_batch = X_batch.float().permute(0, 2, 1).to(device)
                Y_batch_normalized = (Y_batch.float().to(device) - target_mean.to(device)) / target_std.to(device)
                Y_batch = Y_batch.float().to(device)
                
                # Forward pass
                _, predictions = model(X_batch)
                
                # Loss calculation on normalized data
                pred_loss = nn.MSELoss()(predictions, Y_batch_normalized)
                
                # Backpropagation
                optimizer.zero_grad()
                pred_loss.backward()
                optimizer.step()
                
                total_pred_loss += pred_loss.item()
                
                # Unnormalize predictions
                unnormalized_preds = predictions.cpu().detach() * target_std.cpu() + target_mean.cpu()
                all_predictions.append(unnormalized_preds)
                all_true_labels.append(Y_batch.cpu())
            
            # Concatenate predictions and true labels
            all_predictions = torch.cat(all_predictions)
            all_true_labels = torch.cat(all_true_labels)
            
            # Calculate MSE and MAE for each target (SBP and DBP)
            mse_sbp = nn.MSELoss()(all_predictions[:, 0], all_true_labels[:, 0])
            mse_dbp = nn.MSELoss()(all_predictions[:, 1], all_true_labels[:, 1])
            
            mae_sbp = nn.L1Loss()(all_predictions[:, 0], all_true_labels[:, 0])
            mae_dbp = nn.L1Loss()(all_predictions[:, 1], all_true_labels[:, 1])
            
            avg_loss = total_pred_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            # Log metrics
            wandb.log({
                'epoch': epoch,
                'prediction_loss': avg_loss,
                'MSE_SBP': mse_sbp.item(),
                'MSE_DBP': mse_dbp.item(),
                'MAE_SBP': mae_sbp.item(),
                'MAE_DBP': mae_dbp.item()
            })
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'config': config
                }
                torch.save(best_checkpoint, os.path.join(checkpoint_dir, f'best_pred_{checkpoint_file}'))
        
        # Evaluate on test set (similar modifications)
        model.eval()
        test_losses = []
        all_test_predictions = []
        all_test_true_labels = []
        
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.float().permute(0, 2, 1).to(device)
                Y_batch = Y_batch.float().to(device)
                
                # Normalize Y_batch
                Y_batch_normalized = (Y_batch - target_mean.to(device)) / target_std.to(device)
                
                _, predictions = model(X_batch)
                test_loss = nn.MSELoss()(predictions, Y_batch_normalized)
                test_losses.append(test_loss.item())
                
                # Unnormalize predictions
                unnormalized_preds = predictions.cpu() * target_std.cpu() + target_mean.cpu()
                all_test_predictions.append(unnormalized_preds)
                all_test_true_labels.append(Y_batch.cpu())
        
        # Concatenate test predictions and true labels
        all_test_predictions = torch.cat(all_test_predictions)
        all_test_true_labels = torch.cat(all_test_true_labels)
        
        # Calculate test MSE and MAE for each target
        test_mse_sbp = nn.MSELoss()(all_test_predictions[:, 0], all_test_true_labels[:, 0])
        test_mse_dbp = nn.MSELoss()(all_test_predictions[:, 1], all_test_true_labels[:, 1])
        
        test_mae_sbp = nn.L1Loss()(all_test_predictions[:, 0], all_test_true_labels[:, 0])
        test_mae_dbp = nn.L1Loss()(all_test_predictions[:, 1], all_test_true_labels[:, 1])
        
        avg_test_loss = np.mean(test_losses)
        
        # Log and store results
        wandb.log({
            'checkpoint': checkpoint_file,
            'avg_test_loss': avg_test_loss,
            'Test_MSE_SBP': test_mse_sbp.item(),
            'Test_MSE_DBP': test_mse_dbp.item(),
            'Test_MAE_SBP': test_mae_sbp.item(),
            'Test_MAE_DBP': test_mae_dbp.item()
        })
        
        results[checkpoint_file] = {
            'model': model,
            'avg_test_loss': avg_test_loss,
            'test_mse_sbp': test_mse_sbp.item(),
            'test_mse_dbp': test_mse_dbp.item(),
            'test_mae_sbp': test_mae_sbp.item(),
            'test_mae_dbp': test_mae_dbp.item()
        }
                
        wandb.finish()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='grid_search_mlm')
    args = parser.parse_args()

    wandb.login(key="5433ea25cba78fa767acec2d6104b274627cf542")
    wandb.init(project="ppg-processing-grid-search", name=args.run_name)

    # Load datasets
    X_train_emd = torch.load("./X_train_new_emd.pt")
    X_test_emd = torch.load("./X_test_new_emd.pt")
    Y_train = torch.load("./Y_train_dbp_sbp_new.pt")
    Y_test = torch.load("./Y_test_dbp_sbp_new.pt")

    #train_dataset = CustomDataset(X_train_emd['imfs'][:8, :, :], Y_train[:8, :])
    #test_dataset = CustomDataset(X_test_emd['imfs'][:8, :, :], Y_test[:8, :])


    train_dataset = CustomDataset(X_train_emd['imfs'], Y_train)
    test_dataset = CustomDataset(X_test_emd['imfs'], Y_test)

    train_loader = DataLoader(train_dataset, batch_size=2560, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=2560, shuffle=False, collate_fn=collate_fn)

    #grid_search(train_loader, test_loader)

    results = train_prediction_across_checkpoints(train_loader, test_loader)