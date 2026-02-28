import torch
import argparse
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import tensor as tt
import time
import torch.optim as optim
from CDC import *
from torch.distributions.multivariate_normal import MultivariateNormal as tmvnorm
import time
from scipy import stats as scs

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a model with a specified dataset.")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'magic_ecdf')")
parser.add_argument("--epochs", type=str, default=50, help="Number of epochs to train the model.")
parser.add_argument("--cv_seed", type=int, default=0, help="Seed for cross-validation.")
parser.add_argument("--ce_weight", type=float, default=0.2, help="Weight for loss mixture.")
parser.add_argument("--num_timesteps", type=int, default=10, help="num_timesteps/classes in discretaisation of time/classification")
parser.add_argument("--GG_cdc", type=int,default=0, help="Use a Gaussian-Guided CDC; use a Gaussian as terminal distribution (True=1/False=0).")
args = parser.parse_args()

# Use dataset name to construct file paths and variable names
dataset_name = args.dataset
cv_seed = int(args.cv_seed)
csv_path = f"Data/{dataset_name}.csv"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check if the dataset file exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset file '{csv_path}' not found.")

# Load the dataset
X_ecdf = pd.read_csv(csv_path).values.astype(np.float32)

# transform to Gaussian scale
X_ecdf = scs.norm.ppf(X_ecdf.clip(1e-5,1-1e-5)) # for overflow/underflow

# Split into train and test sets
X_ecdf_train, X_ecdf_test, _, _ = train_test_split(X_ecdf, X_ecdf, test_size=0.2, random_state=cv_seed)

# Dataloaders
if dataset_name == 'robocup_train':
    X_train_tensor = torch.tensor(X_ecdf, dtype=torch.float32)
    print('Using full dataset for robocup_train')
else:
    X_train_tensor = torch.tensor(X_ecdf_train, dtype=torch.float32)
train_dataset = TensorDataset(X_train_tensor)

batch_size =  1024  # Adjust batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


ce_weight = torch.tensor(args.ce_weight, dtype=torch.float)
T_max = 3
num_timesteps = int(args.num_timesteps)
num_epochs = int(args.epochs)
print_every = 50
lr = 0.00005
if args.GG_cdc == 1:
    GG_cdc = True
else:
    GG_cdc = False

if GG_cdc:
    # Define OU covariance matrix for Gaussian copula CDC
    corr_mat = torch.corrcoef(X_train_tensor.T).to(device)
    if dataset_name == 'Dry_Bean_ecdf': # ensure p.d.
        corr_mat = corr_mat + 1e-5 * torch.eye(corr_mat.shape[0], device=corr_mat.device)
        corr_mat = corr_mat/(1+1e-5)
        print('towned down corr_mat a bit for the Dry_Bean_ecdf dataset')
    print("Using Gaussian-Guided CDC with OU covariance matrix. Shape: ", corr_mat.shape)

print(f"Training with dataset: {dataset_name}, of shape {X_train_tensor.shape}, epochs: {num_epochs}, num_timesteps: {num_timesteps}, ce_weight: {ce_weight}, GG_cdc: {GG_cdc}, cv_seed: {cv_seed}")



ce_loss_fn = nn.CrossEntropyLoss()
mse_loss_fn = nn.MSELoss()

timesteps = (torch.linspace(tt(-2*T_max).exp(), 1, num_timesteps).log()/-2).__reversed__().to(device)


# define model
model = ResNetCDClassifier(input_dim=X_train_tensor.shape[1],
                           device=device, 
                            num_timesteps=num_timesteps, 
                            time_steps=timesteps, 
                            hidden_dim=512, 
                            depth=6)

if dataset_name == 'Dry_Bean_ecdf':
    model = ResNetCDClassifier_corr(input_dim=X_train_tensor.shape[1],
                           device=device, 
                            num_timesteps=num_timesteps, 
                            time_steps=timesteps, 
                            hidden_dim=512, 
                            depth=6)
    print(f"Using correlated model for {dataset_name}")

if GG_cdc: # add the correlation matrix, it is needed during training
    model.corr_mat = corr_mat
    model.corr_mat_inv = torch.linalg.inv(corr_mat)
    model.corr_mat_chol = torch.linalg.cholesky(corr_mat)
    print('Using corr_mat for:', dataset_name)

# num of params in model
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

optimizer = optim.Adam(model.parameters(), lr=lr)

loss_cum = torch.zeros(3, device=device)
loss_hist = []

training_start_time = time.time()

for epoch in (range(num_epochs+1)):
    for x0, in train_loader:
        optimizer.zero_grad()
        x0 = x0.to(device)
        # Sample random timesteps for each sample in batch
        t_idx = torch.randint(0, num_timesteps, (x0.shape[0],)).to(device)
        
        # Simulate noisy data
        if GG_cdc:
            noise_OU = tmvnorm(loc=torch.zeros_like(x0).to(device),
                                covariance_matrix=corr_mat
                                ).sample().to(device)
        else:
            noise_OU = torch.randn_like(x0).to(device)  # Standard Gaussian noise
        x_t, noise = sample_ou_noised_discrete(x0, t_idx, timesteps, noise=noise_OU)
        x_t.requires_grad_()

        # Forward pass
        logits, denoiser = model(x_t, t_idx, return_score=True)
        # Loss
        ce_loss = ce_loss_fn(logits, t_idx)
        mse_loss = mse_loss_fn(denoiser, noise)
        total_loss = ce_weight * ce_loss + mse_loss

        # Backprop + optimize
        total_loss.backward()
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        with torch.no_grad():
            loss_cum[0] += total_loss
            loss_cum[1] += ce_weight * ce_loss
            loss_cum[2] += mse_loss
            loss_hist.append([total_loss.item(), ce_loss.item(), mse_loss.item()])
            print(f"per --- Epoch {epoch}: Total={loss_cum[0].item():.5f} | w*CE={loss_cum[1].item():.5f} | MSE={loss_cum[2].item():.5f}")
            if epoch % print_every == 0:
                print(f"Epoch {epoch}: Total={loss_cum[0].item()/print_every:.5f} | w*CE={loss_cum[1].item()/print_every:.5f} | MSE={loss_cum[2].item()/print_every:.5f}")
            loss_cum.zero_()
    if epoch % 1000 == 0:
        torch.save(model.state_dict(), f'Model_weights/CDC_{dataset_name}_seed_{args.cv_seed}_iter_{epoch+1}_num_timesteps_{num_timesteps}.pt')
        

#X_ecdf_train = tt(X_ecdf_train).float().to(device)
#X_ecdf_test = tt(X_ecdf_test).float().to(device)
#
#
#ll_train = model.estimate_log_density_ratio(X_ecdf_train).to(device)
#ll_test = model.estimate_log_density_ratio(X_ecdf_test).to(device)
#
#if GG_cdc:
#    print('eval LL with Gaussian-Guided CDC')
#    ll_train_GG = tmvnorm(loc=torch.zeros_like(X_ecdf_train).to(device),
#                                covariance_matrix=corr_mat).log_prob(X_ecdf_train)
#    ll_train_norm = tmvnorm(loc=torch.zeros_like(X_ecdf_train).to(device),
#                                covariance_matrix=torch.eye(X_ecdf_train.shape[1]).to(device)).log_prob(X_ecdf_train)
#    ll_train = ll_train + ll_train_GG - ll_train_norm
#
#    ll_test_GG = tmvnorm(loc=torch.zeros_like(X_ecdf_test).to(device),
#                                covariance_matrix=corr_mat).log_prob(X_ecdf_test)
#    ll_test_norm = tmvnorm(loc=torch.zeros_like(X_ecdf_test).to(device),
#                                covariance_matrix=torch.eye(X_ecdf_test.shape[1]).to(device)).log_prob(X_ecdf_test)
#    ll_test = ll_test + ll_test_GG - ll_test_norm
#
#    print(f"Log-likelihood (with GG rescale) on train set: {ll_train.mean().item():.5f} with std {ll_train.std().item():.5f}")
#    print(f"Log-likelihood (with GG rescale) on test set: {ll_test.mean().item():.5f} with std {ll_test.std().item():.5f}")
#else:
#    print(f"Log-likelihood on train set: {ll_train.mean().item():.5f} with std {ll_train.std().item():.5f}")
#    print(f"Log-likelihood on test set: {ll_test.mean().item():.5f} with std {ll_test.std().item():.5f}")
#
torch.save(model.state_dict(), f'Model_weights/CDC_{dataset_name}_seed_{args.cv_seed}_iter_{epoch+1}_num_timesteps_{num_timesteps}.pt')

# save LL in a file
#np.save(f"Model_samples/CDC/ll_train_{dataset_name}_seed{cv_seed}_ce{args.ce_weight}_epochs{num_epochs}_num_timesteps_{num_timesteps}.npy",ll_train.cpu().detach().numpy())
#np.save(f"Model_samples/CDC/ll_test_{dataset_name}_seed{cv_seed}_ce{args.ce_weight}_epochs{num_epochs}_num_timesteps_{num_timesteps}.npy",ll_test.cpu().detach().numpy())
with torch.no_grad():
            
    ll_train = model.estimate_log_density_ratio(torch.tensor(X_ecdf_train).to(device).float()).to(device)
    ll_test = model.estimate_log_density_ratio(torch.tensor(X_ecdf_test).to(device).float()).to(device)
    
    print(f"{epoch} -------------------- LL train {ll_train.mean().item():.5f} +- {ll_train.std().item():.5f}, LL Test {ll_test.mean().item():.5f} +- {ll_test.std().item():.5f}")
    np.save(f"Model_samples/CDC/ll_train_{dataset_name}_seed{cv_seed}_ce{args.ce_weight}_epochs{num_epochs}_num_timesteps_{num_timesteps}.npy",ll_train.cpu().detach().numpy())
    np.save(f"Model_samples/CDC/ll_test_{dataset_name}_seed{cv_seed}_ce{args.ce_weight}_epochs{num_epochs}_num_timesteps_{num_timesteps}.npy",ll_test.cpu().detach().numpy())



# Sampling
sampling_time= time.time()
sims = (model.sample_with_denoiser(num_samples=1000, return_all=True, exploration=True).cpu().detach().numpy())
print('SAMPLING TIME:',time.time()-sampling_time)
np.save(f"Model_samples/CDC/samples_{dataset_name}_seed{cv_seed}_ce{args.ce_weight}_epochs{args.epochs}_num_timesteps_{num_timesteps}.npy", sims)
print(f"Samples saved to Model_samples/CDC/samples_{dataset_name}_seed{cv_seed}_ce{args.ce_weight}_epochs{args.epochs}_num_timesteps_{num_timesteps}.npy")

print('time taken to train Cdc:', time.time()-training_start_time, 's, dataset:', dataset_name, 'cv_seed:', cv_seed , 'epochs:', args.epochs)
