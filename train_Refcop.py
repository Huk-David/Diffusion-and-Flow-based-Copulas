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
from Ref_copula import *



# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a model with a specified dataset.")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'magic_ecdf')")
parser.add_argument("--epochs", type=str, default=50, help="Number of epochs to train the model.")
parser.add_argument("--cv_seed", type=str, default=0, help="Seed for cross-validation.")
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

# Split into train and test sets
X_ecdf_train, X_ecdf_test, _, _ = train_test_split(X_ecdf, X_ecdf, test_size=0.2, random_state=cv_seed)

# Dataloaders
if dataset_name == 'robocup_train':
    X_train_tensor = torch.tensor(X_ecdf, dtype=torch.float32)
    print('Using full dataset for robocup_train since already split')
else:
    X_train_tensor = torch.tensor(X_ecdf_train, dtype=torch.float32)
train_dataset = TensorDataset(X_train_tensor)

batch_size =  512  # Adjust batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



# Modelling with Reflection Flow Copula

MIN_SUPPORT = 0
MAX_SUPPORT = 1
MAX_TIME = 1.5
DATA_DIMS = X_ecdf_train.shape[1]

print('training velocity field, dataset:', dataset_name, 'cv_seed:', cv_seed , 'epochs:', args.epochs)

training_start_time = time.time()

# training arguments
lr = 1e-4
iterations = int(args.epochs)
print_every = 1000

U_train = tt(X_ecdf_train).to(torch.float32) 
size = U_train.shape[0]

# velocity field model init
vf = Ref_copula(input_dim=DATA_DIMS, time_dim=1, hidden_dim=512, num_layers=6).to(device) 

# init optimizer
optim = torch.optim.Adam(vf.parameters(), lr=lr) 
loss_tracker = tt(0.)

# train
start_time = time.time()
for i in range(int(iterations) + 1):
    # Get a random batch from the loader
    try:
        X_batch = next(train_iter)
    except NameError:
        train_iter = iter(train_loader)
        X_batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        X_batch = next(train_iter)
    
    optim.zero_grad()
    X_0 = X_batch[0]
    V_0 = torch.randn_like(X_0)

    t = MAX_TIME * (torch.rand(X_0.shape[0]).to(device) ** 4)
    t = t.to(device)
    X_0 = X_0.to(device)
    V_0 = V_0.to(device)

    X_t, V_t = simulate_forward(X_0, V_0, t, MIN_SUPPORT, MAX_SUPPORT)

    pred_error = vf(X_t, t) - V_t
    loss = torch.pow(pred_error, 2).mean()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(vf.parameters(), max_norm=1.0)
    optim.step()

    loss_tracker += loss.item()

    if (i+1) % 5 == 0:    
        elapsed = time.time() - start_time
        print('| iter {:6d} | {:5.2f} ms/step | loss {:8.7f} , pred_error {:8.4f}' 
            .format(i+1, elapsed*1000/print_every, loss_tracker.item()/(print_every),pred_error[-1,0].item())) 
        start_time = time.time()
        loss_tracker = tt(0.)
    # log loss
    if (i+1) % print_every == 0:
        torch.save(vf.state_dict(), f'Model_weights/Ref_copula_{dataset_name}_seed_{args.cv_seed}_iter_{i+1}.pt')
        print(f'Iter {i+1} saving model weights for {dataset_name}.')
    
    



print('time taken to train velocity field:', time.time()-training_start_time, 's, dataset:', dataset_name, 'cv_seed:', cv_seed , 'epochs:', args.epochs)
