import pickle
import numpy as np

def adimensionalize(data, data_min, data_max):
  return (2.0 * data - data_min - data_max) / (data_max - data_min)

def dimensionalize(data, data_min, data_max):
  return (data_min + data_max + (data_max - data_min) * data) / 2.0

def read_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return data

def mse(predictions, observations):
    return ((predictions - observations) ** 2).mean()

def nmse(predictions, observations, normalization):
    return ((predictions - observations) ** 2).mean() / (normalization ** 2)

def split_range_into_chunks(num_points, n_chunks):
    points_per_chunk = num_points // n_chunks
    remainder = num_points % n_chunks

    chunks = []
    start = 0
    for i in range(n_chunks):
        chunk_size = points_per_chunk + (1 if i < remainder else 0)
        end = start + chunk_size
        chunks.append(range(start, end))
        start = end

    return chunks