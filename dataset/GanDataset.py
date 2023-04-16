import os
from .GenericDataset import PatchDataset, get_data_info

class GanDataset(PatchDataset):
    def __init__(self, prefix, path, label, num_frames=None):
        # Store the path for the HDF5 file
        path = os.path.join(prefix, path)
        # Create an empty list to store data information
        data_info = get_data_info(path,label,num_frames) 
        super().__init__(data_info)