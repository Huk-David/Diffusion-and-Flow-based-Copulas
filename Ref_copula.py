import torch
import torch.nn as nn

def reflection(x, v, a1, a2):
    """Function used for reflected velocities."""
    x_rescaled = (x - a1)/(a2-a1)
    v_rescaled = v/(a2-a1)
    remainders = (x_rescaled % 1)
    multiplier = (x_rescaled // 1).int()
    mask = (multiplier % 2)
    rescaled_refl_value = mask * (1-remainders) + (1-mask) * remainders
    refl_value = rescaled_refl_value * (a2-a1) + a1
    v_refl =  (a2-a1) * (- mask * v_rescaled + (1-mask) * v_rescaled)
    return refl_value, v_refl

# activation and model
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        return torch.sigmoid(x)*x

# model class
class Ref_copula(nn.Module):
    def __init__(self,
                 input_dim=2,
                 time_dim=1,
                 hidden_dim=256,
                 num_layers=2,
                 act=Swish(),
                 ):
        """
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            hidden_dim (int): Number of hidden units in each layer.
            num_layers (int): Number of hidden layers.
            activation (nn.Module): Activation function to use.
        """
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.act = act

        # Define the layers dynamically
        layers = []
        layers.append(nn.Linear(input_dim+time_dim, hidden_dim))  # Input layer
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))  # Hidden layers
            layers.append(act)
        layers.append(nn.Linear(hidden_dim, input_dim))  # Output layer

        # Store the layers in a Sequential container
        self.model = nn.Sequential(*layers)

    def forward(self, input, t):
        # init
        sz = input.size()
        input = input.view(-1, self.input_dim)
        t = t.view(-1, self.time_dim).float()

        # forward
        t = t.reshape(-1, 1).expand(input.shape[0], 1)        
        h = torch.cat([input, t], dim=1) # concat        
        output = self.model(h) # forward
        
        return output.view(*sz)


def simulate_forward(X_data, V_0, time, min_support=None,max_support=None):
    """
    Simulate forward in time using the specified velocities.
    """
    X_t_unref = X_data + time[:,None] * V_0
    V_t_unref = V_0
    X_t, V_t = reflection(X_t_unref, V_t_unref, min_support, max_support)
    return X_t, V_t
