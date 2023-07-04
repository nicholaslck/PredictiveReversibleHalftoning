import os
import json
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from os.path import join
from glob import glob

from utils.convert import bgr2gray, bgr2lab, bgr2ycrcb, normalize_tensor, tensor_0_1, tensor_0_255

MINI_DATASET_SIZE=1600

class HalftoneVOC2012(Dataset):
    # data range is [-1,1], color image is in BGR format
    def __init__(self, root_dir, data_list, is_mini=False):
        super(HalftoneVOC2012, self).__init__()
        assert root_dir
        assert data_list

        self.inputs = [join(root_dir, x) for x in data_list['inputs']]
        self.labels = [join(root_dir, x) for x in data_list['labels']]
        self.is_mini = is_mini

    @staticmethod
    def load_input(name):
        bgr = cv2.imread(name, cv2.IMREAD_COLOR)
        # transpose data from H,W,C --> C,H,W
        bgr = bgr.transpose((2, 0, 1))
        bgr_t = torch.from_numpy(bgr.astype(np.float32))

        # convert BGR from [0,255] to [0,1]
        bgr_t = tensor_0_1(bgr_t)
        gray_t = bgr2gray(bgr_t)
        ycrcb_t = bgr2ycrcb(bgr_t)
        # lab_t = bgr2lab(bgr_t)

        # convert from [0,1] to [0,255] to [-1, -1]
        img = normalize_tensor(tensor_0_255(bgr_t))
        img_gray = normalize_tensor(tensor_0_255(gray_t))
        img_ycrcb = normalize_tensor(tensor_0_255(ycrcb_t))
        # img_lab = normalize_tensor(tensor_0_255(lab_t))

        return img, img_gray, [img_ycrcb]
 
    @staticmethod
    def load_label(name):
        """Halftone image by Ostromokov method"""
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        # transpose data from H,W --> C,H,W
        img = img[np.newaxis, :, :]
        # to Tensor porject [0, 255] --> [-1, 1]
        img = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
        return img

    def __getitem__(self, index):
        input_data = self.load_input(self.inputs[index])
        label_data = self.load_label(self.labels[index])
        return input_data[0], input_data[1], input_data[2], label_data

    def __len__(self):
        return len(self.inputs) if not self.is_mini else MINI_DATASET_SIZE
    
class HalftoneVOC2012Training(HalftoneVOC2012):
    def __init__(self, root_dir, is_mini=False):
        __map = None
        with open(os.path.join(root_dir, "HalftoneVOC2012.json")) as f:
            __map = json.load(f)
            data_list = __map["train"]
        assert data_list is not None
        super(HalftoneVOC2012Training, self).__init__(root_dir, data_list, is_mini)

class HalftoneVOC2012Validation(HalftoneVOC2012):
    def __init__(self, root_dir, is_mini=False):
        __map = None
        with open(os.path.join(root_dir, "HalftoneVOC2012.json")) as f:
            __map = json.load(f)
            data_list = __map["val"]
        assert data_list is not None
        super(HalftoneVOC2012Validation, self).__init__(root_dir, data_list, is_mini)

class PlainColorTrainingDataset(HalftoneVOC2012):
    def __init__(self, root_dir, is_mini=False):
        __map = None
        with open(os.path.join(root_dir, "special.json")) as f:
            __map = json.load(f)
            data_list = __map["train"]
        assert data_list is not None
        super(PlainColorTrainingDataset, self).__init__(root_dir, data_list, is_mini)

class ColorRampValidationDataset(Dataset):
    def __init__(self):
        super(ColorRampValidationDataset, self).__init__()
        __root_dir = "test/color_ramp/256x256/"
        self.input_names = os.listdir(__root_dir)
        self.inputs = list(map(lambda x: os.path.join(__root_dir, x), self.input_names))
        __label_dir = "ov_halftone/color_ramp/ov"
        self.label_names = os.listdir(__label_dir)
        self.labels = list(map(lambda x: os.path.join(__label_dir, x), self.label_names))

    @staticmethod
    def load_input(name):
        bgr = cv2.imread(name, cv2.IMREAD_COLOR)
        # transpose data from H,W,C --> C,H,W
        bgr = bgr.transpose((2, 0, 1))
        bgr_t = torch.from_numpy(bgr.astype(np.float32))

        # convert BGR from [0,255] to [0,1]
        bgr_t = tensor_0_1(bgr_t)
        gray_t = bgr2gray(bgr_t)
        ycrcb_t = bgr2ycrcb(bgr_t)
        # lab_t = bgr2lab(bgr_t)

        # convert from [0,1] to [0,255] to [-1, -1]
        img = normalize_tensor(tensor_0_255(bgr_t))
        img_gray = normalize_tensor(tensor_0_255(gray_t))
        img_ycrcb = normalize_tensor(tensor_0_255(ycrcb_t))
        # img_lab = normalize_tensor(tensor_0_255(lab_t))

        return img, img_gray, [img_ycrcb]
    
    @staticmethod
    def load_label(name):
        """Halftone image by Ostromokov method"""
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        # transpose data from H,W --> C,H,W
        img = img[np.newaxis, :, :]
        # to Tensor porject [0, 255] --> [-1, 1]
        img = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
        return img
    
    def __getitem__(self, index):
        input_data = self.load_input(self.inputs[index])
        label_data = self.load_label(self.labels[index])
        return input_data[0], input_data[1], input_data[2], label_data

    def __len__(self):
        return len(self.inputs)

    def getitem_name(self, index):
        return self.input_names[index]

class MHXiaHalftoneVOC2012(Dataset):
    def __init__(self, is_training=False):
        root_dir = "dataset/mhxia_C"
        assert os.path.isdir(root_dir)
        if is_training:
            self.inputs = glob(join(root_dir, "Train/target", '*.png'))
        else:
            self.inputs = glob(join(root_dir, "Test/target", '*.png'))

    @staticmethod
    def load_input(name):
        bgr = cv2.imread(name, cv2.IMREAD_COLOR)
        # transpose data from H,W,C --> C,H,W
        bgr = bgr.transpose((2, 0, 1))
        bgr_t = torch.from_numpy(bgr.astype(np.float32))

        # convert BGR from [0,255] to [0,1]
        bgr_t = tensor_0_1(bgr_t)
        gray_t = bgr2gray(bgr_t)
        ycrcb_t = bgr2ycrcb(bgr_t)
        # lab_t = bgr2lab(bgr_t)

        # convert from [0,1] to [0,255] to [-1, -1]
        img = normalize_tensor(tensor_0_255(bgr_t))
        img_gray = normalize_tensor(tensor_0_255(gray_t))
        img_ycrcb = normalize_tensor(tensor_0_255(ycrcb_t))
        # img_lab = normalize_tensor(tensor_0_255(lab_t))

        return img, img_gray, [img_ycrcb]

    @staticmethod
    def load_label(name):
        """Halftone image by Ostromokov method"""
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        # transpose data from H,W --> C,H,W
        img = img[np.newaxis, :, :]
        # to Tensor porject [0, 255] --> [-1, 1]
        img = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
        return img
 
    
    def __getitem__(self, index):
        input_data = self.load_input(self.inputs[index])
        label_data = self.load_label(self.inputs[index])
        return input_data[0], input_data[1], input_data[2], label_data

    def __len__(self):
        return len(self.inputs)
        
class MHXiaHalftoneVOC2012Gray(MHXiaHalftoneVOC2012):
    
    @staticmethod
    def load_input_as_gray(name):
        gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # transpose data from H,W,C --> C,H,W
        bgr = bgr.transpose((2, 0, 1))
        bgr_t = torch.from_numpy(bgr.astype(np.float32))

        # convert BGR from [0,255] to [0,1]
        bgr_t = tensor_0_1(bgr_t)
        gray_t = bgr2gray(bgr_t)
        ycrcb_t = bgr2ycrcb(bgr_t)
        # lab_t = bgr2lab(bgr_t)

        # convert from [0,1] to [0,255] to [-1, -1]
        img = normalize_tensor(tensor_0_255(bgr_t))
        img_gray = normalize_tensor(tensor_0_255(gray_t))
        img_ycrcb = normalize_tensor(tensor_0_255(ycrcb_t))
        # img_lab = normalize_tensor(tensor_0_255(lab_t))

        return img, img_gray, [img_ycrcb]
    
    def __getitem__(self, index):
        input_data = self.load_input_as_gray(self.inputs[index])
        label_data = self.load_label(self.inputs[index])
        return input_data[0], input_data[1], input_data[2], label_data
        

