import os, time, random, hydra
import numpy as np
import pyvista as pv

from omegaconf import DictConfig

import torch as th
import tensorflow as tf
# Set float32 type for training/testing operations.
th.set_default_dtype(th.float32)
tf.keras.backend.set_floatx('float32')
th.set_float32_matmul_precision('high')

from LFLDNets import LFLDNetCell

from utils import *

@hydra.main(version_base = None, config_path = ".", config_name = "config")
def main(cfg: DictConfig):
    # Test indices from the dataset.
    if cfg.test_case == "EP":
        test_indices = list(range(100, 150))
    if cfg.test_case == "CFD":
        test_indices = list(range(25, 32))

    # Model label.
    model_label = "LFLDNets_"
    # Path to dataset with numerical simulations.
    dataset_file = "./data/" + cfg.test_case + ".pkl"
    # Base output folder.
    out_folder = "./output/" + cfg.test_case + "/"

    # Testing samples.
    num_test = len(test_indices)

    # Read dataset.
    dataset = read_pkl(dataset_file)
    # Points.
    if dataset["points"].ndim == 2:
        num_points, num_coords = dataset["points"].shape
    elif dataset["points"].ndim == 3:
        _, num_points, num_coords = dataset["points"].shape
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
    if dataset["points"].ndim == 2:
        points_min = np.array(dataset["points_min"], dtype = np.float32).reshape(1, num_coords)
        points_max = np.array(dataset["points_max"], dtype = np.float32).reshape(1, num_coords)
    elif dataset["points"].ndim == 3:
        points_min = np.array(dataset["points_min"], dtype = np.float32).reshape(1, 1, num_coords)
        points_max = np.array(dataset["points_max"], dtype = np.float32).reshape(1, 1, num_coords)
    points_adim = th.tensor(adimensionalize(dataset["points"].astype(np.float32), points_min, points_max), dtype = th.float32)
    # Import parameters and signals (inputs) + adimensionalization.
    inputs_adim = np.zeros((num_simulations, num_times, 0))
    num_inputs = num_params + num_signals
    if "parameters" in dataset:
        params_min = np.array(dataset["parameters_min"], dtype = np.float32).reshape(1, num_params)
        params_max = np.array(dataset["parameters_max"], dtype = np.float32).reshape(1, num_params)
        params_adim = np.tile(np.expand_dims(adimensionalize(dataset["parameters"].astype(np.float32), params_min, params_max), axis = 1), (1, num_times, 1))
        inputs_adim = np.concatenate((inputs_adim, params_adim), axis = 2)
    if "signals" in dataset:
        signals_min = np.array(dataset["signals_min"], dtype = np.float32).reshape(1, num_signals)
        signals_max = np.array(dataset["signals_max"], dtype = np.float32).reshape(1, num_signals)
        signals_adim = adimensionalize(dataset["signals"].astype(np.float32), signals_min, signals_max)
        inputs_adim = np.concatenate((inputs_adim, signals_adim), axis = 2)
    inputs_adim = th.tensor(inputs_adim, dtype = th.float32)
    # Import outputs (without adimensionalization).
    outputs_min = np.array(dataset["outputs_min"], dtype = np.float32).reshape(1, 1, 1, num_outputs)
    outputs_max = np.array(dataset["outputs_max"], dtype = np.float32).reshape(1, 1, 1, num_outputs)
    outputs = dataset["outputs"].astype(np.float32)
    outputs_adim = adimensionalize(outputs, outputs_min, outputs_max)
    # Mask for homogeneous Dirichlet boundary conditions.
    use_mask = cfg.training.use_mask
    if use_mask:
        mask = th.tensor(dataset["mask"], dtype = th.float32)

    # Instantiate the model.
    model = LFLDNetCell(num_coords, num_inputs, num_outputs,
                        cfg.training.lr,
                        cfg.lfldnet_architecture.N_states, cfg.lfldnet_architecture.N_neu, cfg.lfldnet_architecture.N_hid, cfg.lfldnet_architecture.fourier_mapping_size,
                        th.tensor(outputs_min, dtype = th.float32),
                        th.tensor(outputs_max, dtype = th.float32),
                        use_mask)

    # Testing (one simulation at a time, all time steps, chunks of mesh points, for RAM constraints).
    checkpoint = th.load(cfg.lfldnet_architecture.chk_path, map_location = th.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    chunks = split_range_into_chunks(num_points, cfg.inference.num_chunks)
    volume_mesh = pv.read(out_folder + "/mesh.vtu")
    for idx_s in range(len(test_indices)):
        # Run one numerical simulation for each chunk of mesh points.
        start_time = time.time()
        outputs_NN = np.zeros((1, num_points, num_times, num_outputs))
        for chunk in chunks:
            if use_mask:
                predictions_adim = model([points_adim[chunk, :].unsqueeze(0),
                                          inputs_adim[test_indices[idx_s], ...].unsqueeze(0),
                                          mask[chunk, ...].unsqueeze(0)])
            else:
                predictions_adim = model([points_adim[chunk, :].unsqueeze(0),
                                          inputs_adim[test_indices[idx_s], ...].unsqueeze(0)])
            outputs_NN[0, chunk, :, :] = predictions_adim["outputs"].detach().numpy()

        # Compute testing error for each numerical simulation.
        print('=======================================')
        print(f'Simulation index: {test_indices[idx_s]}')
        print(f"Testing time: {time.time() - start_time:.4f} seconds")
        print(f'Mean square error (adimensional): {mse(outputs_NN[0, ...], outputs_adim[test_indices[idx_s], ...])}')
        outputs_NN = dimensionalize(outputs_NN, outputs_min, outputs_max)
        print(f'Mean square error (dimensional): {mse(outputs_NN[0, ...], outputs[test_indices[idx_s], ...])}')

        # Export all time steps.
        if not os.path.exists(out_folder + str(test_indices[idx_s]) + "/"):
            os.makedirs(out_folder + str(test_indices[idx_s]) + "/")
        for idx_t in range(num_times):
            volume_mesh.point_data['Solution_NN'] = outputs_NN[0, :, idx_t, :]
            volume_mesh.point_data['Solution_numerical'] = outputs[test_indices[idx_s], :, idx_t, :]
            volume_mesh.save(out_folder + str(test_indices[idx_s]) + "/output_" + str(idx_t) + ".vtu")

if __name__ == "__main__":
    main()