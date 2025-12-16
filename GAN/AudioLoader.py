import os
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

class AudioDataset(Dataset):
    def __init__(self, data_dir, target_sample_rate=16000, num_samples=None, transformation=None):

        self.data_dir = Path(data_dir)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.transformation = transformation
        
        extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg']
        self.files = []
        for ext in extensions:
            self.files.extend(list(self.data_dir.rglob(ext)))
            
        if len(self.files) == 0:
            print(f"Warn: No files found in {self.data_dir}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        audio_path = self.files[index]
        
        signal, sr = torchaudio.load(audio_path)
        
        signal = signal.to("cpu")

        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)

        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        if self.num_samples:
            if signal.shape[1] > self.num_samples:
                signal = signal[:, :self.num_samples]
            elif signal.shape[1] < self.num_samples:
                num_missing_samples = self.num_samples - signal.shape[1]
                last_dim_padding = (0, num_missing_samples)
                signal = torch.nn.functional.pad(signal, last_dim_padding)

        if self.transformation:
            signal = self.transformation(signal)

        label = audio_path.stem 
        return signal, label
