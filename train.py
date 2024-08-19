import os, time, random
import numpy as np
import omegaconf

import torch as th
import tensorflow as tf
# Set float32 type for training/testing operations.
th.set_default_dtype(th.float32)
tf.keras.backend.set_floatx('float32')
th.set_float32_matmul_precision('high')

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from ray import train, tune
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import (RayDDPStrategy, RayLightningEnvironment, RayTrainReportCallback, prepare_trainer)
from ray.tune.search.optuna import OptunaSearch

from LFLDNets import LFLDNetCell, DatasetLFLDNet

from utils import *

# Search space for ray hyperparameter tuning.
search_space = {"N_states": tune.choice([50, 100, 150]),
                "N_neu": tune.choice([200, 250, 300, 350, 400]),
                "N_hid": tune.choice([5, 10, 15]),
                "fourier_mapping_size" : tune.choice([25, 50, 75, 100, 125, 150, 175, 200])}

# Get configuration file.
cfg = omegaconf.OmegaConf.load("config.yaml")

# Model label.
model_label = "LFLDNets_"
# Path to dataset with numerical simulations.
dataset_file = "./data/" + cfg.test_case + ".pkl"

# Training and test indices from the dataset.
if cfg.test_case == "EP":
    train_indices = list(range(0, 100))
    test_indices = list(range(100, 150))
if cfg.test_case == "CFD":
    train_indices = list(range(0, 25))
    test_indices = list(range(25, 32))

# Set the random seed.
th.manual_seed(cfg.training.seed)
random.seed(cfg.training.seed)
np.random.seed(cfg.training.seed)

# Training/Testing samples.
num_train = len(train_indices)
num_test = len(test_indices)

# Parameters for the dataloader.
dataloader = {'batch_size': cfg.training.batch_size, 'shuffle': False}

# Read dataset.
dataset = read_pkl(dataset_file)
# Points.
num_points, num_coords = dataset["points"].shape
# Input parameters.
if "parameters" in dataset:
    _, num_params = dataset["parameters"].shape
else:
    num_params = 0
# Input signals.
if "signals" in dataset:
    _, _, num_signals = dataset["signals"].shape
else:
    num_signals = 0
# Outputs.
num_simulations, num_points, num_times, num_outputs = dataset["outputs"].shape
# Import times.
times = dataset["times"]
num_times = len(times)
times_adim = times / times[-1]
# Import mesh points + adimensionalization.
points_min = np.array(dataset["points_min"], dtype = np.float32).reshape(1, num_coords)
points_max = np.array(dataset["points_max"], dtype = np.float32).reshape(1, num_coords)
points_adim = th.tensor(adimensionalize(dataset["points"].astype(np.float32), points_min, points_max), requires_grad = True, dtype = th.float32)
# Import parameters and signals (inputs) + adimensionalization.
inputs_adim = np.zeros((num_simulations, num_times, 0), dtype = np.float32)
num_inputs = num_params + num_signals
if "parameters" in dataset:
    params_min = np.array(dataset["parameters_min"], dtype = np.float32).reshape(1, num_params)
    params_max = np.array(dataset["parameters_max"], dtype = np.float32).reshape(1, num_params)
    params_adim = np.tile(np.expand_dims(adimensionalize(dataset["parameters"].astype(np.float32), params_min, params_max), axis = 1), (1, num_times, 1))
    inputs_adim = np.concatenate((inputs_adim, params_adim), axis = 2)
if "signals" in dataset:
    signals_min = np.array(dataset["signals_min"], dtype = np.float32).reshape(1, 1, num_signals)
    signals_max = np.array(dataset["signals_max"], dtype = np.float32).reshape(1, 1, num_signals)
    signals_adim = adimensionalize(dataset["signals"].astype(np.float32), signals_min, signals_max)
    inputs_adim = np.concatenate((inputs_adim, signals_adim), axis = 2)
inputs_adim = th.tensor(inputs_adim, dtype = th.float32)
# Import outputs (without adimensionalization).
outputs_min = np.array(dataset["outputs_min"], dtype = np.float32).reshape(1, 1, 1, num_outputs)
outputs_max = np.array(dataset["outputs_max"], dtype = np.float32).reshape(1, 1, 1, num_outputs)
outputs_adim = adimensionalize(dataset["outputs"].astype(np.float32), outputs_min, outputs_max)

# Mask for homogeneous Dirichlet boundary conditions.
use_mask = cfg.training.use_mask
if use_mask:
    mask = th.tensor(dataset["mask"], dtype = th.float32)

# Create dataloader.
training_set = DatasetLFLDNet(points_adim,
                              inputs_adim[train_indices, :, :], outputs_adim[train_indices, :, :, :],
                              cfg.training.sampled_mesh_points,
                              mask if use_mask else None)
validation_set = DatasetLFLDNet(points_adim,
                                inputs_adim[test_indices, :, :], outputs_adim[test_indices, :, :, :],
                                cfg.training.sampled_mesh_points,
                                mask if use_mask else None)
training_generator = th.utils.data.DataLoader(training_set, **dataloader)
validation_generator = th.utils.data.DataLoader(validation_set, **dataloader)

def train_valid_func(config):
    # Instantiate the model.
    model = LFLDNetCell(num_coords, num_inputs, num_outputs,
                        cfg.training.lr,
                        config["N_states"], config["N_neu"], config["N_hid"], config["fourier_mapping_size"],
                        th.tensor(outputs_min, dtype = th.float32),
                        th.tensor(outputs_max, dtype = th.float32),
                        use_mask)

    # Define trainer.
    logger = TensorBoardLogger(save_dir = os.getcwd(), version = "", name = "")
    trainer = L.Trainer(devices = "auto", accelerator = "auto", strategy = RayDDPStrategy(),
                        precision = "32-true",
                        callbacks = [RayTrainReportCallback()],
                        plugins = [RayLightningEnvironment()],
                        logger = logger,
                        # No gradient clipping.
                        gradient_clip_val = None, gradient_clip_algorithm = "value",
                        # No gradient accumulation.
                        accumulate_grad_batches = 1,
                        min_epochs = 1, max_epochs = cfg.training.epochs_adam,
                        max_time = {"days": 2}, enable_progress_bar = False,
                        log_every_n_steps = cfg.training.output_frequency,
                        check_val_every_n_epoch = 1)
    trainer = prepare_trainer(trainer)

    # Training.
    trainer.fit(model = model,
                train_dataloaders = training_generator,
                val_dataloaders = validation_generator,
                ckpt_path = "last")

def hpo():
    scaling_config = ScalingConfig(num_workers = cfg.hpo.num_workers, use_gpu = cfg.hpo.use_gpu)
    
    checkpoint_config = CheckpointConfig(num_to_keep = 2,
                                         checkpoint_score_attribute = "valid_loss",
                                         checkpoint_score_order = "min")
    run_config = RunConfig(checkpoint_config = checkpoint_config, storage_path = cfg.hpo.base_folder, name = model_label + cfg.test_case + "_seed" + str(cfg.training.seed))

    ray_trainer = TorchTrainer(train_valid_func, scaling_config = scaling_config, run_config = run_config)

    if cfg.hpo.restart:
        tuner = tune.Tuner.restore(cfg.hpo.base_folder + model_label + cfg.test_case + "_seed" + str(cfg.training.seed), trainable = ray_trainer)
    else:
        tuner = tune.Tuner(ray_trainer,
                           param_space = {"train_loop_config": search_space},
                           tune_config = tune.TuneConfig(search_alg = OptunaSearch(), metric = "valid_loss", mode = "min", num_samples = cfg.hpo.num_samples))

    return tuner.fit()

def single_execution():
    # Checkpoint folder.
    checkpoint_folder = model_label + str(cfg.lfldnet_architecture.N_states) +"s_dyn" + \
                                      str(cfg.lfldnet_architecture.N_neu) + "n_rec" + \
                                      str(cfg.lfldnet_architecture.N_neu) + "n" + \
                                      str(cfg.lfldnet_architecture.N_hid) + "l_" + \
                                      str(cfg.lfldnet_architecture.fourier_mapping_size) + "f_" + \
                                      str(cfg.training.sampled_mesh_points) + "sp_" + \
                                      str(cfg.training.epochs_adam) + "adam_" + \
                                      f"{cfg.training.lr:.4f}" + "lr_" + \
                                      str(cfg.training.seed) + "seed_" + \
                                      cfg.test_case

    # Instantiate the model.
    model = LFLDNetCell(num_coords, num_inputs, num_outputs,
                        cfg.training.lr,
                        cfg.lfldnet_architecture.N_states, cfg.lfldnet_architecture.N_neu, cfg.lfldnet_architecture.N_hid, cfg.lfldnet_architecture.fourier_mapping_size,
                        th.tensor(outputs_min, dtype = th.float32),
                        th.tensor(outputs_max, dtype = th.float32),
                        use_mask)

    # Define trainer.
    checkpoint_callback = ModelCheckpoint(monitor = "valid_loss", save_last = True, save_top_k = 3)
    logger = TensorBoardLogger(save_dir = os.getcwd(), version = checkpoint_folder, name = "NNs")
    trainer = L.Trainer(devices = "auto", accelerator = "auto", strategy = "auto",
                        precision = "32-true",
                        callbacks = [checkpoint_callback],
                        logger = logger,
                        # No gradient clipping.
                        gradient_clip_val = None, gradient_clip_algorithm = "value",
                        # No gradient accumulation.
                        accumulate_grad_batches = 1,
                        min_epochs = 1, max_epochs = cfg.training.epochs_adam,
                        max_time = {"days": 2}, enable_progress_bar = True,
                        log_every_n_steps = cfg.training.output_frequency,
                        check_val_every_n_epoch = cfg.training.output_frequency)

    # Training.
    start_time = time.time()
    trainer.fit(model = model,
                train_dataloaders = training_generator,
                val_dataloaders = validation_generator,
                ckpt_path = "last")
    print(f"Training time: {time.time() - start_time:.4f} seconds")

def hpo_with_ray():
    # Hyperparameter tuning.
    results = hpo()
    print("Best configuration selected by ray:", results.get_best_result(metric = "valid_loss", mode = "min").config)

def main():
    if cfg.hpo.active:
        hpo_with_ray()
    else:
        single_execution()

if __name__ == "__main__":
    main()
