import torch
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import tensor as tt
import time
from Ref_copula import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


from tqdm import tqdm



parser = argparse.ArgumentParser(description="Sample a model for a specified dataset.")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'magic_ecdf')")
args = parser.parse_args()
csv_path = f"Data/{args.dataset}.csv"

# Load the dataset
X_ecdf = pd.read_csv(csv_path).values.astype(np.float32)

time_per_samples = []

for cv_seed in range(10):
    # Split into train and test sets
    X_ecdf_train, X_ecdf_test, _, _ = train_test_split(X_ecdf, X_ecdf, test_size=0.2, random_state=cv_seed)

    # Dataloaders
    X_train_tensor = torch.tensor(X_ecdf_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_ecdf_test, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor)
    test_dataset = TensorDataset(X_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Load the Reflection Flow Copula

    MIN_SUPPORT = 0
    MAX_SUPPORT = 1
    MAX_TIME = 1.5
    DATA_DIMS = X_ecdf_train.shape[1]

            
    vf = Ref_copula(input_dim=DATA_DIMS, time_dim=1, hidden_dim=512, num_layers=6).to(device) 
    vf.load_state_dict(torch.load(f'Model_weights/Ref_copula/Ref_copula_{args.dataset}_seed_{cv_seed}_iter_100000.pt'))
    print('velocity field loaded')

    # Sample from the model by simulating the trajectory of the Reflection flow
    batch_size = 30000 
    n_sim_steps = 50
    T = torch.linspace(MAX_TIME, 0, n_sim_steps) # sample times
    T = T.to(device=device)
    

    start_time = time.time()

    def simulate_trajectory(timesteps, data_dim=2):
        
        X_T = (MAX_SUPPORT - MIN_SUPPORT) * torch.rand((batch_size, data_dim), dtype=torch.float32, device=device) + MIN_SUPPORT
        with torch.no_grad():
            x_curr_list = [X_T.detach().cpu()]
            for idx in tqdm(range(len(timesteps)-1)):
                x_curr = x_curr_list[-1].to(device)
                curr_time = timesteps[idx]
                next_time = timesteps[idx+1]
                time_delta = next_time - curr_time
                curr_time_vec = curr_time * torch.ones(size=(x_curr.shape[0],),device=x_curr.device)
                vf_pred = vf(x_curr, curr_time_vec)
                x_next = x_curr + time_delta * vf_pred
                
                x_next = reflection(x_next, vf_pred, MIN_SUPPORT, MAX_SUPPORT)[0]
                x_curr_list.append(x_next.detach().cpu())
        return torch.stack(x_curr_list)

    x_traj = simulate_trajectory(T, data_dim=DATA_DIMS)


    print(f'Simulation time, dataset {args.dataset}, seed {cv_seed}:', time.time() - start_time)
    time_per_samples.append( time.time() - start_time )
    # Save the simulated data
    simulated_data = x_traj[-1].numpy()
    simulated_data = pd.DataFrame(simulated_data, columns=[f'col_{i}' for i in range(simulated_data.shape[1])])
    simulated_data.to_csv(f'Model_samples/Ref_copula/{args.dataset}_simulated_seed_{cv_seed}.csv', index=False)
    print(f"Simulated data saved to Model_samples/Ref_copula/{args.dataset}_simulated_seed_{cv_seed}.csv")


print(f"Average time per sample for dataset {args.dataset}: {np.mean(time_per_samples)} seconds, std: {np.std(time_per_samples)} seconds")