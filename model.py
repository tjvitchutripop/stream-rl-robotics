import torch
import torch.nn as nn
from cbp_linear import CBPLinear
from streaming_drl.sparse_init import sparse_init

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)


class ActorMeanLN(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=256):
        super(ActorMeanLN, self).__init__()
        self.fc1 = nn.Linear(n_obs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, n_actions)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.activation = nn.Tanh()
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.activation(self.ln1(self.fc1(x)))
        x = self.activation(self.ln2(self.fc2(x)))
        x = self.activation(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        return x
    
class CriticLN(nn.Module):
    def __init__(self, n_obs=11, hidden_size=256):
        super(CriticLN, self).__init__()
        self.fc1 = nn.Linear(n_obs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.activation = nn.Tanh()
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.activation(self.ln1(self.fc1(x)))
        x = self.activation(self.ln2(self.fc2(x)))
        x = self.activation(self.ln3(self.fc3(x)))
        x = self.value(x)
        return x

class ActorMeanDeep(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=256):
        super(ActorMeanDeep, self).__init__()
        self.fc1 = nn.Linear(n_obs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, n_actions)
        self.activation1 = nn.Tanh()
        self.activation2 = nn.Tanh()
        self.activation3 = nn.Tanh()
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.activation3(self.fc3(x))
        x = self.fc4(x)
        return x
    
class CriticDeep(nn.Module):
    def __init__(self, n_obs=11, hidden_size=256):
        super(CriticDeep, self).__init__()
        self.fc1 = nn.Linear(n_obs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)
        self.activation1 = nn.Tanh()
        self.activation2 = nn.Tanh()
        self.activation3 = nn.Tanh()
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.activation3(self.fc3(x))
        x = self.value(x)
        return x

class ActorMean(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=256):
        super(ActorMean, self).__init__()
        self.fc1 = nn.Linear(n_obs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, n_actions)
        self.activation1 = nn.Tanh()
        self.activation2 = nn.Tanh()
        self.activation3 = nn.Tanh()
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.activation3(self.fc3(x))
        x = self.fc4(x)
        return x
    
class Critic(nn.Module):
    def __init__(self, n_obs=11, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_obs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)
        self.activation1 = nn.Tanh()
        self.activation2 = nn.Tanh()
        self.activation3 = nn.Tanh()
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.activation3(self.fc3(x))
        x = self.value(x)
        return x

class ActorMeanCBP(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=256, replacement_rate=1e-5, maturity_threshold=1000, decay_rate=0.99):
        super(ActorMeanCBP, self).__init__()
        self.fc1 = nn.Linear(n_obs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, n_actions)
        self.activation = nn.Tanh()
        self.apply(initialize_weights)

        self.cbp1 = CBPLinear(in_layer=self.fc1, out_layer=self.fc2, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, act_type="tanh", decay_rate=decay_rate)
        self.cbp2 = CBPLinear(in_layer=self.fc2, out_layer=self.fc3, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, act_type="tanh", decay_rate=decay_rate)
        self.cbp3 = CBPLinear(in_layer=self.fc3, out_layer=self.fc4, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, act_type="tanh", decay_rate=decay_rate)

    def forward(self, x):
        x = self.cbp1(self.activation(self.fc1(x)))
        x = self.cbp2(self.activation(self.fc2(x)))
        x = self.cbp3(self.activation(self.fc3(x)))
        x = self.fc4(x)
        return x
    
class CriticCBP(nn.Module):
    def __init__(self, n_obs=11, hidden_size=128, replacement_rate=1e-5, maturity_threshold=1000, decay_rate=0.99):
        super(CriticCBP, self).__init__()
        self.fc1 = nn.Linear(n_obs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()
        self.apply(initialize_weights)

        self.cbp1 = CBPLinear(in_layer=self.fc1, out_layer=self.fc2, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, act_type="tanh", decay_rate=decay_rate)
        self.cbp2 = CBPLinear(in_layer=self.fc2, out_layer=self.fc3, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, act_type="tanh", decay_rate=decay_rate)
        self.cbp3 = CBPLinear(in_layer=self.fc3, out_layer=self.value, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, act_type="tanh", decay_rate=decay_rate)

    def forward(self, x):
        x = self.cbp1(self.activation(self.fc1(x)))
        x = self.cbp2(self.activation(self.fc2(x)))
        x = self.cbp3(self.activation(self.fc3(x)))
        x = self.value(x)
        return x
    
class LinearWithLN(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)
        self.ln = torch.nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.ln(self.fc(x))