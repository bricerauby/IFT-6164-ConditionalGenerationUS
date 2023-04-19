import torch
import h5py
import numpy as np
from .GenericDataset import get_data_info


class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self, path, num_frames,reduced_len=None, key="patches") -> None:
        self.key = key
        self.data_info = get_data_info(path, label=1, num_frames=num_frames,key=key)
        if reduced_len:
            self.reduced_len = reduced_len
        else : 
            self.reduced_len = len(self.data_info)

    def __getitem__(self, index):
        path, patch_idx, frame_idx, label = self.data_info[index]
        # Load the specified frame from the patch in the HDF5 file
        with h5py.File(path, 'r') as h5f:
            patches_dataset = h5f[self.key]
            frame = patches_dataset[frame_idx, :, :, patch_idx]
            frame = frame.T
            frame = (frame - frame.min()) / (frame.max() - frame.min())
        # Convert the frame to a PyTorch tensor
        frame_tensor = torch.tensor(frame, dtype=torch.float32)
        frame_tensor = frame_tensor.unsqueeze(0)
        # Return the frame tensor
        return frame_tensor, label

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.reduced_len


class GanSampleDataset(torch.utils.data.Dataset):
    def __init__(self, path, num_frames,reduced_len=None, key="patches") -> None:
        self.key = key
        self.path = path

        if reduced_len:
            self.reduced_len = reduced_len
        else : 
            with h5py.File(self.path, 'r') as h5f:
                patches_dataset = h5f[self.key]
                self.reduced_len = patches_dataset.shape[0]

    def __getitem__(self, index):
        # Load the specified frame from the patch in the HDF5 file
        with h5py.File(self.path, 'r') as h5f:
            patches_dataset = h5f[self.key]
            frame = patches_dataset[index, :, :, ]
            frame = (frame - frame.min()) / (frame.max() - frame.min())
        # Convert the frame to a PyTorch tensor
        frame_tensor = torch.tensor(frame, dtype=torch.float32)
        frame_tensor = frame_tensor.unsqueeze(0)
        # Return the frame tensor
        return frame_tensor, 1

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.reduced_len