from torch import nn
import numpy as np
import torch

class Classifier(nn.Module):
    def __init__(self, para_conv_layers, input_size=28):
        super(Classifier, self).__init__()
        self.layers        = nn.ModuleList()
        self.para          = para_conv_layers
        self.accuracy      = 0
        self.learning_coef = 0
        self.time_training = 0

        in_channels = 1  # Image en niveaux de gris donc 1 channel d'entré
        for params in self.para:
            out_channels, kernel_size, dropout = params
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)) # une couche de convolution
            self.layers.append(nn.BatchNorm2d(out_channels))                                              # une couche de normalization par batch
            self.layers.append(nn.ReLU())                                                                 # une fonction relu
            self.layers.append(nn.MaxPool2d(2))                                                           # Adding pooling layer to reduce dimensions
            self.layers.append(nn.Dropout(dropout))                                                       # une dropout pour la couche
            in_channels = out_channels


        # Pour la couche linéaire en fin de réseau on a besoin de savoir on a combien de sortie de la dernière couche de conv pour 
        # savoir il faut mettre combien d'entré à cette couche et c'est très chiant à faire proprement
        self._initialize_fc_layer(input_size, in_channels)

    def _initialize_fc_layer(self, input_size, in_channels):
        # on fait passer un tenser dans notre reseau et on regarde de quelle forme il resort en gros
        dummy_input = torch.zeros(1, 1, input_size, input_size)
        with torch.no_grad():
            for layer in self.layers:
                dummy_input = layer(dummy_input)
        n_out = dummy_input.numel()  # nombre d'entré pour la couche linéaire
        self.fc1 = nn.Linear(n_out, 25)



    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)  # on applatie la sortie de la couche de conv pour faire l'entré de la couche linéaire
        x = torch.relu(self.fc1(x))
        return x

    def count_parameters(self):
        # fonction qui permet de compter le nombre de para qui change pendant l'entraîenemnt
        return sum(p.numel() for p in self.parameters() if p.requires_grad)