import torch
import h5py
import numpy as np

def get_data_info(path,label,num_frames):
    """
    Extract data information from the given HDF5 file.

    Args:
        path (str): The path to the HDF5 file.
        label (int): The label associated with the data.
        num_frames (int, optional): The number of frames to process. If None, all frames in the dataset will be processed.

    Returns:
        list: A list of tuples containing the path, patch index, frame index, and label for each frame in the dataset.
    """
    with h5py.File(path, 'r') as h5f:
        patches_dataset = h5f['patches']
        # Get the total number of patches and the number of frames per patch
        num_patches = patches_dataset.shape[-1]
        if num_frames is None:
            num_frames = patches_dataset.shape[0]
        data_info =[]
        # Iterate through each patch index and frame index
        for patch_idx in range(num_patches):
            for frame_idx in range(num_frames):
                data_info.append((path, patch_idx, frame_idx, label)) 
        return data_info
        
class PatchDataset(torch.utils.data.Dataset):
    """
    A Abstract subclass of dataset for handling patches.
    """

    def __init__(self, data_info):
        self.data_info = data_info
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
            frame = np.sqrt(frame[0] ** 2 + frame[1] ** 2)
            frame = (frame - frame.min()) / (frame.max() - frame.min())
        # Convert the frame to a PyTorch tensor
        frame_tensor = torch.tensor(frame, dtype=torch.float32)
        frame_tensor = frame_tensor.unsqueeze(0)
        # Return the frame tensor

        return frame_tensor, label