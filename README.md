# Diffusion and Flow-based Copulas: <br> Forgetting and Remembering Dependencies

This repository contains an implementations of the two proposed models in our ICLR 26 paper ([https://arxiv.org/abs/2509.19707](https://openreview.net/forum?id=YrX77XRgku)), namely the Classification-Diffusion Copula (CDC) and Reflection Copula. These models handle high-dimensional dependency modeling using diffusion processes and flow-based techniques for density estimation and sampling. These are the first copula models able to scale to complex high-dimensional (d=1024) and multimodal dependencies.

Overview
--------
The paper introduces two processes that *forget* inter-variable dependencies while preserving dimension-wise marginals, defining valid copulas at all times. Models then *remember* these dependencies, provably recovering the true copula.

CDC (Classification-Diffusion Copula): Uses a Ornstein-Uhlenbeck process on Gaussian-scale copula data, diffusing the dependence. A classifier estimates the density ratio to recover the copula, and a denoiser enables sampling. The process forgets dependencies exponentially fast, converging to independence.

Reflection Copula (HFC): Defines a reflection velocity field on the unit hypercube, simulating a flow that forgets dependencies while preserving uniform marginals. A neural network estimates the velocity field, and ODE sampling recovers the copula. The process converges to independence as well.

Files
-----
File              | Description                         
------------------|-------------------------------------
[CDC.py](CDC.py)           | CDC architecture                     
[HFC.py](HFC.py)           | Reflection Copula architecture      
[Sample_HFC.py](Sample_HFC.py)    | HFC sampling script                  
[train_HFC.py](train_HFC.py)     | HFC training                         
[train_CDC.py](train_CDC.py)     | CDC training                         


## Using the `CDC` for Density Estimation and Sampling

### 1. **CDC Initialization**
You need to initialize the model with the required parameters:
- `input_dim`: Dimensionality of the input data.
- `device`: Device to run the model (`'cuda'` or `'cpu'`).
- `num_timesteps`: Number of timesteps for the OU process.
- `hidden_dim`: Hidden layer size in the backbone.
- `depth`: Number of residual blocks in the backbone.
- `time_steps`: Tensor of time values for the OU process.
- `corr_mat`: Optional correlation matrix for correlated Gaussian noise.

```python
model = ResNetCDClassifier_corr(
    input_dim=your_input_dim,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    num_timesteps=11,
    hidden_dim=32,
    depth=2,
    time_steps=torch.linspace(0, 1, 11),
    corr_mat=None  # Optional for correlated version
)
```
### 2. **CDC Density Estimation**
To estimate the log density ratio at a specific timestep, use the `estimate_log_density_ratio` function. Provide the noised input `x_t` (a tensor of shape `(B, D)`, where `B` is the batch size and `D` is the feature dimension) and optionally specify the timestep index `t_idx` (default is `0`). The default `0` corresponds to the copula log-density, following Proposition 3 in the paper.

```python
x_t = torch.randn((batch_size, your_input_dim))  # Input data
log_copula = model.estimate_log_density_ratio(x_t, t_idx=0)
print(log_copula)  # Output: (B,) tensor of copula log densities 
```
### 3. **CDC Sampling**
To generate samples from the model, use the `sample_with_denoiser` function. This function starts with Gaussian noise and iteratively denoises it using the model to produce samples.

```python
samples = model.sample_with_denoiser(num_samples=100)
print(samples)  # Output: (100, input_dim) tensor of copula samples
```

## Using the `Reflection Copula` for fast Sampling

### 1. **Reflection Copula Initialization**

You need to initialize the model with the required parameters:
- `input_dim`: Dimensionality of the input data (`DATA_DIMS`, default=number of features in the dataset).
- `device`: Device to run the model (`'cuda'` if GPU is available, otherwise `'cpu'`, default='cpu').
- `time_dim`: Dimensionality of the time variable (default=1).
- `hidden_dim`: Hidden layer size in the backbone (default=512).
- `num_layers`: Number of residual blocks in the backbone (default=6).
- `MIN_SUPPORT`: Minimum support for the data domain (default=0).
- `MAX_SUPPORT`: Maximum support for the data domain (default=1).
- `MAX_TIME`: Maximum time value for the simulation (default=1.5).
```python
# Reflection copula model initialization
vf = Ref_copula(
    input_dim=DATA_DIMS,  # Dimensionality of the input data
    time_dim=1,           # Dimensionality of the time variable
    hidden_dim=512,       # Number of hidden units in each layer
    num_layers=6          # Number of layers in the model
).to(device)              # Move the model to the specified device (CPU/GPU)
```

### 2. **Reflection Copula Prediction**
To output a prediction from the Reflection Copula model, pass the input data and time data through the model using its forward method.

```python
input_data = torch.tensor([[0.5, 0.3], [0.2, 0.8]], device=device)  # Shape: (batch_size, input_dim)
time_data = torch.tensor([0.1, 0.2], device=device)                # Shape: (batch_size, time_dim)
predictions = model(input_data, time_data)
print(predictions)  # Output: (batch_size, input_dim) tensor of predictions
```

### 3. **Reflection Copula Sampling**
To generate samples from the Reflection Copula model, use the `Sample_Ref_copula.py` file. This script simulates the ODE of the Reflection Copula model and saves the generated samples to a CSV file. A minimal example by just using the `simulate_trajectory` function from that file is below.

```python
vf.load_state_dict(torch.load('Model_weights.pt'))       # Load trained model
n_sim_steps = 50                                         # Number of simulation steps
T = torch.linspace(MAX_TIME, 0, n_sim_steps).to(device)  # Time steps
x_traj = simulate_trajectory(T, data_dim=2)              # ODE trajectory, shape: (n_sim_steps, batch_size, data_dim)
samples = x_traj[-1]                                     # Take the last step of the trajectory as samples
print(samples)                                           # Shape: (batch_size, data_dim)
```

# Datasets and Model Weights
For datasets, generated samples, and trained model weights, please contact the first author at the following address: `David.Huk@warwick.ac.uk`.

# Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@inproceedings{
huk2026diffusion,
title={Diffusion and Flow-based Copulas: Forgetting and Remembering Dependencies},
author={David Huk and Theodoros Damoulas},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=YrX77XRgku}
}
```