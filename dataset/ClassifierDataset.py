import os
import torch
import h5py
import numpy as np 

class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, prefix, pathMb, pathNoMb, num_frames=None):
        # Store the path for the HDF5 file
        self.MBdataPath = os.path.join(prefix,pathMb)
        self.noMBdataPath =  os.path.join(prefix,pathNoMb)
        # Create an empty list to store data information
        self.data_info = []
        for path, label in zip([self.noMBdataPath, self.MBdataPath], [0, 1]):
            # Open the HDF5 file and retrieve the 'patches' dataset
            # keeping in mind that Matlab 
            with h5py.File(path, 'r') as h5f:
                patches_dataset = h5f['patches']
                
                # Get the total number of patches and the number of frames per patch
                num_patches = patches_dataset.shape[-1]
                if num_frames is None:
                    num_frames = patches_dataset.shape[0]

                # Iterate through each patch index and frame index
                for patch_idx in range(num_patches):
                    for frame_idx in range(num_frames):
                        self.data_info.append((path, patch_idx, frame_idx, label))

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data_info)

    def __getitem__(self, index):
        # Get the path, patch index, frame index and label for the specified index
        path, patch_idx, frame_idx, label = self.data_info[index]
        # Load the specified frame from the patch in the HDF5 file
        with h5py.File(path, 'r') as h5f:
            patches_dataset = h5f['patches']
            frame = patches_dataset[frame_idx, :, :, :, patch_idx]
            frame = frame.T
            frame = np.sqrt(frame[0]**2 + frame[1]**2)
            frame = (frame - frame.min())/(frame.max()-frame.min())
        # Convert the frame to a PyTorch tensor
        frame_tensor = torch.tensor(frame, dtype=torch.float32)
        frame_tensor = frame_tensor.unsqueeze(0)
        # Return the frame tensor
        return frame_tensor,label