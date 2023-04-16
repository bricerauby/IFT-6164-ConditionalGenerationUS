import os
from .GenericDataset import PatchDataset, get_data_info

class ClassifierDataset(PatchDataset):
    def __init__(self, prefix, pathMb, pathNoMb, num_frames=None):
        # Store the path for the HDF5 file
        MBdataPath = os.path.join(prefix, pathMb)
        noMBdataPath = os.path.join(prefix, pathNoMb)
        # Create an empty list to store data information
        data_info = []
        for path, label in zip([noMBdataPath, MBdataPath], [0, 1]):
            data_info += get_data_info(path,label,num_frames)
        super().__init__(data_info)

if __name__ == "__main__" :
    dataPrefix = "/mnt/f/IFT6164/data"
    train_dataset = ClassifierDataset(dataPrefix, 'trainMB.h5', 'trainNoMB.h5', num_frames=16)
    val_dataset = ClassifierDataset(dataPrefix, 'testMB.h5', 'testNoMB.h5', num_frames=16)
    image,label = train_dataset[0]
    print(image.shape)
