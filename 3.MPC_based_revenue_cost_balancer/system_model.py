import torch
import torch.nn as nn

class SystemModel(nn.Module):
    """Neural network model for system dynamics prediction.
    
    This model predicts the next state of the system based on current state and control inputs.
    It uses a fully connected neural network architecture with configurable layers and activations.
    """
    def __init__(self,   
                    input_shape,
                    output_shape,
                    hidden_units=[256, 256, 256], 
                    activation='relu',
                    output_activation='sigmoid'
        ):
        super(SystemModel, self).__init__()
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._hidden_units = hidden_units
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(self._input_shape, self._hidden_units[0]))
        
        for i in range(len(self._hidden_units)-1):
            self.layers.append(nn.Linear(self._hidden_units[i], self._hidden_units[i+1]))
        
        self.layers.append(nn.Linear(self._hidden_units[-1], self._output_shape))
        
        self.activation = self._get_activation(activation)
        self.output_activation = self._get_activation(output_activation)
        
        self._initialize_weights()
    
    def _get_activation(self, activation_name):
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'none': nn.Identity()
        }
        return activations.get(activation_name.lower(), nn.ReLU())
    
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'embedding' in name:
                param.requires_grad = False
    
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.activation(self.layers[i](x))
        
        x = self.output_activation(self.layers[-1](x))
        
        return x.squeeze(1)

