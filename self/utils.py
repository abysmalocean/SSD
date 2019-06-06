import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT

def create_data_lists(voc07_path, voc12_path, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images
    , and save these to files

    :param voc07_path: path to the 'VOC2007' folder
    :param voc12_path: path to the 'VOC2012' folder
    :param output_folder: folder where the JSONs must be saved
    
    """
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images  = list()
    train_objects = list()
    nObjects      = 0 

    # For Training image
    for path in [voc07_path, voc12_path]:

        # Find IDs of images in training data
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()
        
        for id in ids:
            # Parse annotation's XML file
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            if len(objects) == 0:
                continue
            n_objects += len(objects)
            


def transform(image, boxes, labels, difficulties, split):
    """
    Apply the transformations above.

    :@param image:  image, a PIL Image
    :@param boxes:  bounding boxes in boundary coordinates, 
                   a tensor of dimensions (n_objects, 4)
    :@param labels: labels of objects, 
                   a tensor of dimensions (n_objects)
    :@param difficulties: difficulties of detection 
                         of these objects, a tensor of dimensions (n_objects)
    :@param split:  one of 'TRAIN' or 'TEST', 
                   since different sets of transformations are applied
    :@return:       transformed image, 
                   transformed bounding box coordinates, 
                   transformed labels, 
                   transformed difficulties
    """
    assert split in {'TRAIN', 'TEST'}
    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # TODO: implement the transformations

    return new_image, new_boxes, new_labels, new_difficulties



