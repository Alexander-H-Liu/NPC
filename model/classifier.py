import torch
import torch.nn as nn

class CLF(nn.Module):
    def __init__(self, feat_dim, num_layers, hidden_size, n_class):
        """ Simple MLP Classifier (num_layers=0 for linear classifier) """
        super(CLF, self).__init__()
        input_size = feat_dim
        self.num_layers = num_layers
        if num_layers>0:
            input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
            output_sizes = [hidden_size] * num_layers
            self.layers = nn.ModuleList(
                [nn.Linear(in_features=in_size, out_features=out_size)
                for (in_size, out_size) in zip(input_sizes, output_sizes)])
            self.output = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()
        else:
            self.output = nn.Linear(input_size, n_class)

    def forward(self, input_feature):
        if self.num_layers>0:
            for layer in self.layers:
                input_feature = self.relu(layer(input_feature))
        pred = self.output(input_feature)
        return pred
