import os, sys, random
import numpy as np

import torch as th
from torch.nn.modules.module import Module
from torch.nn import Linear, LayerNorm
from torch.nn.functional import softmax
import torch.nn.functional as F

from ncps.wirings import AutoNCP 
from ncps.torch import CfC

import lightning as L

from utils import *

class DatasetLFLDNet(th.utils.data.Dataset):
    """
    Class to define the dataloader for Liquid Fourier Latent Dynamics Networks.
    """

    # Initialization.
    def __init__(self, points, inputs, outputs, num_sampled_points, mask = None):
        # Input/output fields.
        self.points = points
        self.inputs = inputs
        self.outputs = outputs

        # Space sampling.
        self.num_points = self.points.shape[self.points.ndim - 2]
        self.num_sampled_points = num_sampled_points
        self.idx_points = random.sample(range(0, self.num_points), self.num_sampled_points)

        # Mask for homogeneous Dirichlet boundary conditions.
        self.mask = mask

        # Counter for space re-sampling.
        self.counter = 0

    # Total number of simulations (len acts on the first dimension of the input tensor).
    def __len__(self):
        return len(self.outputs)

    # Get one sample.
    # This function is called for each batch, for each single item in the batch (index is a scalar, not a vector).
    def __getitem__(self, index):
        # Space sub-sampling for mesh points (performed at the beginning of each epoch).
        self.counter += 1
        if self.counter % (len(self.outputs)) == 0:
            self.counter = 0
            self.idx_points = random.sample(range(0, self.num_points), self.num_sampled_points)
        self.sampled_points = self.points[self.idx_points, :]

        if self.mask is None:
            return self.sampled_points, self.inputs[index, ...], self.outputs[index, ...][self.idx_points, ...]
        else:
            return self.sampled_points, self.inputs[index, ...], self.outputs[index, ...][self.idx_points, ...], self.mask[self.idx_points, ...]

class MLP(Module):
    """
    Cell representing a generic feedforward fully-connected neural network.
    """

    def __init__(self, in_feats, out_feats, latent_space, n_h_layers, normalize_output = True):
        super().__init__()

        self.input = Linear(in_feats, latent_space, bias = True)
        self.output = Linear(latent_space, out_feats, bias = True)
        self.n_h_layers = n_h_layers
        self.hidden_layers = th.nn.ModuleList()

        for i in range(self.n_h_layers):
            self.hidden_layers.append(Linear(latent_space, latent_space, bias = True))

        self.normalize_output = normalize_output
        if self.normalize_output:
            self.norm_out = LayerNorm(out_feats)

    def forward(self, inp):
        f = self.input(inp)
        f = F.gelu(f)

        for i in range(self.n_h_layers):
            f = self.hidden_layers[i](f)
            f = F.gelu(f)

        f = self.output(f)

        if self.normalize_output:
            f = self.norm_out(f)

        return f

class FourierEmbedding(Module):
    """
    Cell representing a generic Fourier embedding (i.e. a 2D matrix representing an encoding).
    """

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.encoding = Linear(in_feats, out_feats, bias = False)

    def forward(self, inp):
        return self.encoding(inp)

class LFLDNetCell(L.LightningModule):
    """
    Lightning cell representing a Liquid Fourier Latent Dynamics Network.
    """

    def __init__(self, num_coords, num_inputs, num_outputs,
                 learning_rate,
                 N_states, N_neu, N_hid, fourier_mapping_size,
                 outputs_min, outputs_max,
                 use_mask = False):
        super().__init__()

        self.N_coords = num_coords
        self.N_inputs = num_inputs
        self.N_outputs = num_outputs

        self.lr = learning_rate

        self.N_states = N_states
        self.N_neu = N_neu
        self.N_hid = N_hid

        # Fourier encoding.
        self.fourier_mapping_size = fourier_mapping_size
        self.pi = th.acos(th.Tensor([-1]))
        self.B = FourierEmbedding(self.N_coords, self.fourier_mapping_size)

        self.outputs_min = outputs_min
        self.outputs_max = outputs_max

        self.use_mask = use_mask

        # Liquid dynamic network.
        self.wiring = AutoNCP(self.N_neu, self.N_states)
        self.NN_dyn = CfC(self.N_inputs, self.wiring, batch_first = True, return_sequences = True, mixed_memory = True)

        # Feedforward fully-connected reconstruction network.
        self.NN_rec = MLP(self.N_states + 2 * self.fourier_mapping_size, self.N_outputs, self.N_neu, self.N_hid, normalize_output = False)

    def on_epoch_start(self):
        self.trainer.accelerator.setup()

    def forward(self, x):
        if self.use_mask:
            points, inputs, mask = x
        else:
            points, inputs = x

        batch_size = inputs.shape[0]
        num_points = points.shape[1]
        num_times = inputs.shape[1]

        points = points.to(self.device)
        inputs = inputs.to(self.device)

        # [batch, times, states].
        self.S = th.zeros((batch_size, 1, self.N_states)).to(self.device)

        # Liquid dynamic network.
        # [batch, times, inputs] -> [batch, times, states].
        self.S = self.NN_dyn(inputs)[0]
            
        # [batch, times, states] -> [batch, points, times, states].
        self.S = th.tile(self.S.unsqueeze(1), (1, num_points, 1, 1))

        # Fourier encoding.
        self.pi = self.pi.to(self.device)
        # [batch, points, coordinates] -> [batch, points, 2 * fourier].
        points_projected = self.B(2. * self.pi * points)
        points = th.cat([th.sin(points_projected), th.cos(points_projected)], dim = -1)

        # [batch, points, 2 * fourier] -> [batch, points, times, 2 * fourier].
        points = th.tile(points.unsqueeze(2), (1, 1, num_times, 1))

        # Feedforward fully-connected reconstruction network.
        # [batch, points, times, state + 2 * fourier] -> [batch, points, times, outputs].
        outputs = self.NN_rec(th.cat((self.S, points), dim = -1))

        # Apply mask for homogeneous Dirichlet boundary conditions.
        if self.use_mask:
            self.outputs_min = self.outputs_min.to(self.device)
            self.outputs_max = self.outputs_max.to(self.device)
            mask = mask.to(self.device)
            mask = th.tile(mask.unsqueeze(2), (1, 1, num_times, self.N_outputs))
            outputs = adimensionalize(mask * dimensionalize(outputs, self.outputs_min, self.outputs_max), self.outputs_min, self.outputs_max)

        # Define predictions.
        predictions = {"outputs" : outputs}

        return predictions

    def training_step(self, batch, batch_idx):
        # Preprocessing.
        if self.use_mask:
            points, inputs, outputs_exact, mask = batch
        else:
            points, inputs, outputs_exact = batch

        # Loss functions (data-driven).
        if self.use_mask:
            predictions = self.forward([points, inputs, mask])
        else:
            predictions = self.forward([points, inputs])
        loss_data = mse(predictions["outputs"], outputs_exact)

        # Logging.
        self.log("train_loss", loss_data)

        # Return loss (passed to the optimizer for training).
        return loss_data

    def validation_step(self, batch, batch_idx):
        # Preprocessing.
        if self.use_mask:
            points, inputs, outputs_exact, mask = batch
        else:
            points, inputs, outputs_exact = batch

        # Loss function (data-driven).
        if self.use_mask:
            predictions = self.forward([points, inputs, mask])
        else:
            predictions = self.forward([points, inputs])
        loss_data = mse(predictions["outputs"], outputs_exact)

        # Logging.
        self.log("valid_loss", loss_data)

    def configure_optimizers(self):
        optimizer = th.optim.Adam(list(self.NN_dyn.parameters()) + list(self.NN_rec.parameters()) + list(self.B.parameters()),
                                  lr = self.lr)
        
        return optimizer