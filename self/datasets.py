import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform

class PascalVOCDataset(Dataset):
    """
    A PTorch Dataset clas to be used in a Pytorch DataLoader to create batch
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder    = data_folder
        self.keep_difficult = keep_difficult

        # read the data files
        with open(os.path.join(data_folder, self.split+'_image.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split+'_objects.json'), 'r') as j:
            self.images = json.load(j)
        assert len(self.images) == len(self.objects)
        
    