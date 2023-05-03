import os
import h5py
import torch
import numpy as np
from .GenericDataset import PatchDataset, get_data_info


class GanDataset(PatchDataset):
    def __init__(self, prefix, path, label, num_frames=None, reduced_len=None):
        # Store the path for the HDF5 file
        path = os.path.join(prefix, path)
        self.file  = h5py.File(path, 'r')
        data_info = get_data_info(path, label, num_frames)
        super().__init__(data_info, reduced_len)


class SimuDataset(torch.utils.data.Dataset):
    def __init__(self, path, num_frames):
        dataset = h5py.File(path, 'r')

        self.groups = list(dataset.keys())
        # Get the total number of simu
        num_simu = len(self.groups)
        if num_frames is None:
            num_frames = dataset[self.groups[0]]['CorrMap'].shape[0]
        self.data_info = []
        # Iterate through each patch index and frame index
        for patch_idx in range(num_simu):
            for frame_idx in range(num_frames):
                self.data_info.append((path, patch_idx, frame_idx))
        self.dataset = dataset

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data_info)

    def __getitem__(self, index):
        # Get the path, patch index, frame index and label for the specified index
        path, group_idx, frame_idx = self.data_info[index]
        # Load the specified frame from the patch in the HDF5 file

        patches_dataset = self.dataset[self.groups[group_idx]]['CorrMap']
        frame = patches_dataset[frame_idx, :, :, :]
        frame = patches_dataset[frame_idx, :, :, :]
        frame = frame.T
        abs_frame = np.sqrt(frame[0] ** 2 + frame[1] ** 2)
        if self.real:
            frame = (abs_frame - abs_frame.min()) / (abs_frame.max() - abs_frame.min())
            # Convert the frame to a PyTorch tensor
            frame_tensor = torch.tensor(frame, dtype=torch.float32)
            frame_tensor=frame_tensor.unsqueeze(0)
        else:
            frame = (frame - abs_frame.min()) / (abs_frame.max() - abs_frame.min())
            # Convert the frame to a PyTorch tensor
            frame_tensor = torch.tensor(frame, dtype=torch.float32)
        # Return the frame tensor
        return frame_tensor