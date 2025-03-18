"""
The dataset preprocessing is based on DeepAR
https://arxiv.org/pdf/1704.04110.pdf
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch import Tensor
import os, os.path
import urllib.response
import zipfile
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchaudio.datasets import SPEECHCOMMANDS
import os
import urllib.request
import tarfile
import shutil
import torch.utils.data as data
from scipy import integrate
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from scipy.stats import t as t_dist


class SinusoidalDataset(Dataset):
    def __init__(self, num_samples, seq_length=100, num_features=1, 
                 freq_min=10, freq_max=500, num_classes=100, noise=0,
                 add_outlier=0, outlier_factor=3, seed=42):
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)

        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_features = num_features
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_classes = num_classes
        self.noise = noise
        self.add_outlier = add_outlier
        self.outlier_factor = outlier_factor

    def __len__(self):
        # TODO-FIX: len is not being used(?)
        return self.num_samples

    def __getitem__(self, idx):
        np.random.seed(314) # TODO: remove
        idx = 0 # TODO: remove
        # Generate frequencies for each class
        frequencies = [i for i in range(self.freq_min, self.freq_max + 1, 
                                          (self.freq_max - self.freq_min) // self.num_classes)]
        freq = frequencies[idx % self.num_classes]

        t = np.linspace(0, 1, self.seq_length)
        signal = 0.5 * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))
        signal += 0.1 * np.random.randn(self.seq_length)  # initial noise

        # Add a non-linearly increasing or decreasing trend
        trend = np.linspace(-0.5, 0.5, self.seq_length)
        if np.random.rand() < 0.5:
            trend = np.square(trend)
        else:
            trend = -np.square(trend)
        signal += trend

        # Add more complex patterns to the signal
        signal += 0.2 * np.sin(4 * np.pi * freq * t) + 0.1 * np.sin(8 * np.pi * freq * t)

        if self.noise > 0.:
            # signal += self.noise * np.random.randn(self.seq_length)
            signal += self.noise * t_dist.rvs(2.01, size=self.seq_length)
        
        if self.add_outlier > 0:
            std = np.std(signal)
            lower_bound = int(self.seq_length * 0.8)
            upper_bound = int(self.seq_length * 0.95)
            # Ensure at least one index is available
            if upper_bound <= lower_bound:
                lower_bound = self.seq_length - 1
                upper_bound = self.seq_length
            # If the number of outliers is larger than the available indices, limit it
            available_indices = np.arange(lower_bound, upper_bound)
            num_outliers = min(self.add_outlier, len(available_indices))
            outlier_indices = np.random.choice(available_indices, size=num_outliers, replace=False)
            for idx_outlier in outlier_indices:
                sign = np.random.choice([-1, 1])
                signal[idx_outlier] = signal[idx_outlier] + sign * self.outlier_factor * std

        # Create label and sample dictionary
        label = frequencies.index(freq)
        sample = {'input': torch.tensor(signal, dtype=torch.float).view(-1, self.num_features),
                  'label': label}

        return sample
    
class SinusoidalDatasetLong(Dataset):
    def __init__(self, num_samples, seq_length=100, num_features=1, freq_min=10, freq_max=500, num_classes=20):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_features = num_features
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate sinusoidal signals with noise and trend
        frequencies = [i for i in range(self.freq_min, self.freq_max+1, (self.freq_max-self.freq_min)//self.num_classes)]
        freq = frequencies[idx % self.num_classes]

        t = np.linspace(0, 1, self.seq_length)
        signal = 0.5 * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)) + 0.1 * np.random.randn(self.seq_length)

        # Add a non-linearly increasing or decreasing trend
        trend = np.linspace(-0.5, 0.5, self.seq_length)
        if np.random.rand() < 0.5:
            trend = np.square(trend)
        else:
            trend = -np.square(trend) 
        signal += trend

        # Add more complex patterns to the signal
        signal += 0.2 * np.sin(4 * np.pi * freq * t) + 0.1 * np.sin(8 * np.pi * freq * t)

        # Add more noise to the signal
        signal += 0.1 * np.random.randn(self.seq_length)

        signal2 = 0.5 * np.sin(2 * np.pi * (freq / 5) * t + np.random.uniform(0, 2*np.pi)) + 0.1 * np.random.randn(self.seq_length)
        signal2 += 0.2 * np.sin(4 * np.pi * (freq / 5) * t) + 0.1 * np.sin(8 * np.pi * (freq / 5) * t)
        signal2 += 0.1 * np.random.randn(self.seq_length)

        signal[250:] = signal2[250:]

        label = frequencies.index(freq)
        sample = {'input': torch.tensor(signal, dtype=torch.float).view(-1, self.num_features), 'label': label}

        return sample
    
