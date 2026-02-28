import torch
import torch.nn as nn
from torch import tensor as tt
import torch.nn.functional as F
import scipy.stats as scs
from ddpm_unet import *

### Helper functions ###

def sample_ou_noised_discrete(x0, t_idx, timesteps, noise=None):
    """
    Simulate OU noise at given timestep indices.
    x0: (B, D) base data
    t_idx: (B,) integer time indices (0 ... num_timesteps-1)
    timesteps: (num_timesteps,) tensor of time values
    noise: (B, D) optional noise tensor, if None will sample standard Gaussian noise
    """
    t = timesteps[t_idx].unsqueeze(1)  # (B, 1)
    exp_term = torch.exp(-t)           # OU decay
    std_term = torch.sqrt(1 - torch.exp(-2 * t))  # OU variance term
    if noise is None:
        noise = torch.randn_like(x0)       # standard Gaussian noise
    x_t = exp_term * x0 + std_term * noise # OU expression (B, D)

    return x_t, noise

def gaussianize_marginals(X):
    """
    Transform each marginal of the dataset X from data scale to a standard normal distribution.
    Parameters:
    X: ndarray of shape (n_samples, n_features)
        The input dataset.
    Returns:
    X_gauss: ndarray of shape (n_samples, n_features)
        The dataset with each marginal transformed from data scale to a standard normal distribution.
    """
    X_gauss = torch.zeros_like(X)
    n = X.shape[0]
    for d in range(X.shape[1]):
        ranks = torch.argsort(torch.argsort(X[:, d]))
        uniform = (ranks + 1).float() / (n + 1)
        gaussianized = torch.tensor(scs.norm.ppf(uniform.cpu()), device=X.device)
        X_gauss[:, d] = gaussianized
    return X_gauss

### Classification-Diffusion copula ###

# activation 
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        return torch.sigmoid(x)*x
    
# Deep Net block
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.activation = Swish()  

    def forward(self, x):
        return self.activation(x + self.net(x))

# ResNet backbone for MLP
class ResNetMLPBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, depth=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.resblocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(depth)])

    def forward(self, x):
        x = self.input_proj(x)
        return self.resblocks(x)

# ResNet-based classifier for time prediction

class ResNetCDClassifier_corr(nn.Module):
    def __init__(self, input_dim, device, num_timesteps=11, hidden_dim=32, depth=2, time_steps=None, backbone= None, corr_mat=None):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.device = device
        self.corr_mat = corr_mat
        self.corr_mat_inv = torch.linalg.inv(corr_mat) if corr_mat is not None else None
        self.corr_mat_chol = torch.linalg.cholesky(self.corr_mat).to(self.device) if self.corr_mat is not None else None
        if backbone is None:
            self.backbone = ResNetMLPBackbone(input_dim=input_dim, hidden_dim=hidden_dim, depth=depth).to(self.device)
        else:
            self.backbone = backbone.to(self.device)
        self.classifier_head = nn.Linear(hidden_dim, num_timesteps).to(self.device)

    def forward(self, x_t, t_idx, return_score=True):
        """
        x_t: (B, D) OU input = a*obs+b*noise (vector form)
        t_idx: (B,) integer timestep labels (0 ... T)
        return_score: if True, return grad_x log p(t_k | x_t) - log p(t_N | x_t)
        """
        
        x_t.to(self.device).requires_grad_(return_score)  # for gradient tracking if needed

        # Compute logits for time prediction
        if self.backbone is None:
            features = self.backbone(x_t)  # (B, hidden)
        else:
            features = self.backbone(x_t)  # (B, hidden)
        logits = self.classifier_head(features)  # (B, num_timesteps) in R
        log_probs = F.log_softmax(logits, dim=1).to(self.device)  # log p(t_k | x_t) log([0,1]) in R-
        # Compute the denoiser from the gradients
        if return_score:
            log_p_tk = log_probs[:, -1]  # Final T_max=T_k prob (B,)
            log_p_ts = torch.gather(log_probs, 1, t_idx.view(-1, 1)).squeeze(1)  # Prob of querried times s (B,)
            F_ = log_p_ts - log_p_tk
            grad_F = torch.autograd.grad(
                outputs=F_,
                inputs=x_t,
                grad_outputs=torch.ones_like(F_),
                retain_graph=True,
                create_graph=True
            )[0]
            scale = (1 - (-2 * self.time_steps[t_idx]).exp()).sqrt().view(-1, 1) # Scale for OU process
            if self.corr_mat is not None: # Apply correlation matrix to get the full expression for grad_log_p_tk
                grad_log_p_t = grad_F - x_t @ self.corr_mat_inv.T  # ∇_z log p_s = grad log (prob_Ts/prob_Tk)-x_t @ Σ^(-1)
                denoiser = -scale * (grad_log_p_t @ self.corr_mat.T)
            else: # Use identity correlation
                grad_log_p_t = grad_F - x_t  # (B, D)
                denoiser = -scale * grad_log_p_t  # (B, D)
            return logits, denoiser
        else:
            return logits, None
    
    def estimate_log_density_ratio(self, x_t, t_idx=0):
        """
        Estimate log density ratio log p(t_0 | x) - log p(t_N | x)
        x_t: (B, D)
        Returns: log_density_ratio: (B,)
        """
        self.eval()
        if self.backbone is None:
            features = self.backbone(x_t)
        else:
            features = self.backbone(x_t)
        logits = self.classifier_head(features)  # (B, T)
        log_probs = torch.log_softmax(logits, dim=1)       # log p(t_k | x)
        log_ratio = log_probs[:, t_idx] - log_probs[:, -1]     # log p(t_0) - log p(t_N)
        if self.corr_mat is not None:
            # remove the corerlated Gaussian
            corr_gaussian = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=torch.zeros(x_t.shape[1], device=self.device),
                covariance_matrix=self.corr_mat
            ).log_prob(x_t)
            simple_gaussian = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=torch.zeros(x_t.shape[1], device=self.device),
                covariance_matrix=torch.eye(x_t.shape[1], device=self.device)
            ).log_prob(x_t)
            log_ratio += (corr_gaussian - simple_gaussian)
        return log_ratio
    
    def return_prob_ratio_copula(self, x_t):
        """
        Return a binary classification of whether x_t is from t_0 or t_N.
        x_t: (B, D)
        Returns: logits: (B, 2) - logits for class 0 (t_0) and class 1 (t_N)
        """
        log_ratio = self.estimate_log_density_ratio(x_t)  # (B,)
        prob = log_ratio.exp() / (1 + log_ratio.exp())  # Convert to probability
        return prob

    def sample_with_denoiser(self, x_t=None, num_samples=25, return_all=False, exploration=False, clip=6):
        """
        Sample from the model given noised input x_t.
        Returns: sampled outputs (B, D)
        """
        self.eval()
        if return_all:
            x_t_list = 333 * torch.ones(self.num_timesteps, num_samples, self.input_dim, device=self.device).to(self.device)
            # Sample standard Gaussian noise if no input provided
            if x_t is None: 
                x_t = torch.randn((num_samples, self.input_dim), device=self.device).to(self.device)  # (B, D)
                if self.corr_mat is not None: # Apply correlation matrix to noise
                    x_t = x_t @ self.corr_mat_chol.T
            x_t_list[-1] = x_t  # Store initial noise
            alphas = (2 * (self.time_steps[:-1] - self.time_steps[1:])).exp().to(self.device)
            scales = (1 - (-2 * self.time_steps[1:]).exp()).sqrt().to(self.device)

            for t_idx in (reversed(range(1,self.num_timesteps))):
                # Forward pass to get denoiser
                #print(f"Sampling at timestep {t_idx} with input shape {x_t.shape}")
                t_idx_tensor = tt(t_idx).repeat(x_t.shape[0]).to(self.device)  # (B,)
                log_probs, denoiser = self.forward(x_t, t_idx_tensor, return_score=True)
                alpha = tt(2*(self.time_steps[t_idx-1] - self.time_steps[t_idx])).exp().to(self.device)  # (B,)
                scale = (1 - (-2 * self.time_steps[t_idx]).exp()).sqrt().view(-1, 1).to(self.device)
                # Apply denoiser to move towards clean data 
                x_t = (1/alpha.sqrt()) * (x_t - ((1-alpha)/(scale) * denoiser)).to(self.device)  # (B, D) 
                x_t = x_t.clamp(-clip, clip).to(self.device)  # Clamp to avoid exploding values
                x_t.detach_()  # Detach to avoid accumulating gradients

                if exploration:
                    if t_idx >1:
                        noise = torch.randn_like(x_t, device=self.device)
                        if self.corr_mat is not None: # Apply correlation matrix to noise
                            noise = noise @ self.corr_mat_chol.T
                        x_t += noise * (1-alpha).sqrt()
                x_t_list[t_idx-1] = x_t  # Store the sampled output at this timestep
            
            return x_t_list
        else:
            # Sample standard Gaussian noise if no input provided
            if x_t is None: 
                x_t = torch.randn((num_samples, self.input_dim), device=self.device).to(self.device)  # (B, D)
                if self.corr_mat is not None: # Apply correlation matrix to noise
                    x_t = x_t @ self.corr_mat_chol.T
            alphas = (2 * (self.time_steps[:-1] - self.time_steps[1:])).exp().to(self.device)
            scales = (1 - (-2 * self.time_steps[1:]).exp()).sqrt().to(self.device)

            for t_idx in (reversed(range(1,self.num_timesteps))):
                # Forward pass to get denoiser
                #print(f"Sampling at timestep {t_idx} with input shape {x_t.shape}")
                t_idx_tensor = tt(t_idx).repeat(x_t.shape[0]).to(self.device)  # (B,)
                log_probs, denoiser = self.forward(x_t, t_idx_tensor, return_score=True)
                # Apply denoiser to move towards clean data 
                x_t = (1/alphas[t_idx-1].sqrt()) * (x_t - ((1-alphas[t_idx-1])/(scales[t_idx-1]) * denoiser)).to(self.device)  # (B, D) 
                x_t = x_t.clamp(-clip, clip).to(self.device)  # Clamp to avoid exploding values

                x_t.detach_()  # Detach to avoid accumulating gradients

                if exploration:
                    if t_idx >1:
                        noise = torch.randn_like(x_t, device=self.device)
                        if self.corr_mat is not None: # Apply correlation matrix to noise
                            noise = noise @ self.corr_mat_chol.T
                        x_t +=  noise * (1-alphas[t_idx-1]).sqrt() # Add exploration noise
            return x_t
        
