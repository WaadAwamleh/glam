from os import path as osp
import random
import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from pathlib import Path

def generate_random_homography(h, w):
    random.seed(23)
    angle = np.random.uniform(-5, 5)  # rotation angle
    scale = np.random.uniform(0.9, 1.1)  # scale factor
    tx, ty = np.random.uniform(-0.1 * w, 0.1 * w), np.random.uniform(-0.1 * h, 0.1 * h)  # translation
    shear = np.random.uniform(-0.1, 0.1)  # shear factor

    # Rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotation_matrix = np.vstack([rotation_matrix, [0, 0, 1]])

    # Translation matrix
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

    # Shearing matrix
    shear_matrix = np.array([
        [1, shear, 0],
        [shear, 1, 0],
        [0, 0, 1]
    ])
    homography = np.matmul(translation_matrix, np.matmul(rotation_matrix, shear_matrix))
    return homography

def apply_homography(img, homography):
    transformed_img = cv2.warpPerspective(np.uint8(img), homography, (img.shape[1], img.shape[0]))
    transformed_img = transformed_img 
    #print(np.max(transformed_img))
    #transformed_img = transformed_img[:, :, np.newaxis]
    return transformed_img

def crop(img, size):

    img = img.copy()
    if len(img.shape) ==2:
        img = img[:, :, np.newaxis]

    h,w = img.shape[:2]
    pad_w = 0
    pad_h = 0
    if w < size:
        pad_w = np.uint16((size - w)/2)
    if h < size:
        pad_h = np.uint16((size - h)/2)
    img_pad = cv2.copyMakeBorder(img,pad_h,pad_h,pad_w,pad_w,cv2.BORDER_CONSTANT,value=[0, 0, 0])
    if len(img_pad.shape) == 2:
        img_pad = img_pad[:, :, np.newaxis]
    x1 = w // 2 - size // 2
    y1 = h // 2 - size// 2
    img_pad = img_pad[y1:y1 + size, x1:x1 + size, :]
    return img_pad

class SyntheticDataset(Dataset):
    def __init__(self, dir, train=True):
        self.training = train
        self.root_dir = Path(dir)  
        self.list_original_images = [f for f in self.root_dir.rglob('*/*.jpg')] #*healthy/
        print(len(self.list_original_images))
        if train:
            self.list_original_images = self.list_original_images[:int(0.8 * len(self.list_original_images))]
        else:
            self.list_original_images = self.list_original_images[int(0.8 * len(self.list_original_images)):]

    def __len__(self):
        return len(self.list_original_images)

    def __getitem__(self, idx):
        random.seed(23)
        name_image = self.list_original_images[idx]
        path_original_image = os.path.join(self.root_dir, name_image)
        image_full = cv2.imread(path_original_image)
        image = crop(image_full, size = 720)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        g_i = generate_random_homography(h, w)
        g_i_prime = generate_random_homography(h, w)
        I_i = apply_homography(image, g_i)
        I_i_norm = I_i/ (np.max(I_i))
        I_i_prime = apply_homography(image, g_i_prime)
        I_i_prime_norm = I_i_prime/ (np.max(I_i_prime))
        H_prime = np.matmul(g_i_prime, np.linalg.inv(g_i))
        I_i = torch.from_numpy(I_i).unsqueeze(0).float()
        I_i_prime = torch.from_numpy(I_i_prime).unsqueeze(0).float()
        output = {'image1': I_i,
                  'image2': I_i_prime,
                  'image1_normed': torch.Tensor(I_i_norm.astype(np.float32)).unsqueeze(0),
                  'image2_normed': torch.Tensor(I_i_prime_norm.astype(np.float32)).unsqueeze(0),
                  'H1_to_2': torch.from_numpy(H_prime).float()}
        return output

"""
def center_crop(img, size):
    '''
    Get the center crop of the input image
    Args:
        img: input image [HxWx3]
        size: size of the center crop (tuple)
    Output:
        img_pad: center crop
        x, y: coordinates of the crop
    '''

    if not isinstance(size, tuple):
        size = (size, size)

    img = img.copy()
    h, w = img.shape[:2]

    pad_w = 0
    pad_h = 0
    if w < size[1]:
        pad_w = np.uint16((size[1] - w) / 2)
    if h < size[0]:
        pad_h = np.uint16((size[0] - h) / 2)
    img_pad = cv2.copyMakeBorder(img,
                                 pad_h,
                                 pad_h,
                                 pad_w,
                                 pad_w,
                                 cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
    x1 = w // 2 - size[1] // 2
    y1 = h // 2 - size[0] // 2

    img_pad = img_pad[y1:y1 + size[0], x1:x1 + size[1], :]

    return img_pad


class SyntheticDataset(Dataset):
    def __init__(self, cfg, train=True):
        '''
         path - path to directory containing images
                size - (H, W)
                mask - boolean if a mask of the foreground of the umages needs to be computed
                augmentation - dictionnary from config giving info on augmentation
    outputs:
                batch - image N x H x W x 1  intensity scale 0-255
                if mask is True
                also mask N x H x W x 1

        '''

        self.cfg = cfg
        self.seed = cfg['training']['seed']
        self.training = train
        if train:
            self.root_dir = os.path.join(cfg['training']['TRAIN_DIR'])
            if cfg['training']['train_list'] != ' ':
                self.list_original_images = []
                if not os.path.isfile(cfg['training']['train_list']):
                    raise ValueError('The path to train_list you indicated does not exist !')
                self.list_original_images = open(cfg['training']['train_list']).read().splitlines()
            else:
                self.list_original_images = [f for f in os.listdir(self.root_dir) if
                                             f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.ppm'))]
        else:
            # we are evaluating
            self.root_dir = os.path.join(cfg['validation']['VAL_DIR'])
            if cfg['validation']['val_list'] != ' ':
                if not os.path.isfile(cfg['validation']['validation_list']):
                    raise ValueError('The path to validation_list you indicated does not exist !')
                self.list_original_images = []
                self.list_original_images = open(cfg['validation']['val_list']).read().splitlines()
            else:
                self.list_original_images = [f for f in os.listdir(self.root_dir) if
                                             f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.ppm'))]

    def __len__(self):
        return len(self.list_original_images)

    def __getitem__(self, idx):

        name_image = self.list_original_images[idx]
        path_original_image = os.path.join(self.root_dir, name_image)

        # reads the image
        try:
            image_full = cv2.imread(path_original_image)

            # crops it to the desired size
            if self.training:
                image = center_crop(image_full, (self.cfg['training']['image_size_h'], self.cfg['training']['image_size_w']))
            else:
                image = center_crop(image_full,
                                    (self.cfg['validation']['image_size_h'], self.cfg['validation']['image_size_w']))

            # apply correct preprocessing
            if self.cfg['augmentation']['use_green_channel']:
                image = image[:, :, 1]
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            '''
            if self.mask:
                mask = (image < 230) & (image > 25)
            '''

            # sample homography and creates image1
            h1 = homography_sampling(image.shape, self.cfg['sample_homography'], seed=self.seed * (idx + 1))
            image1 = cv2.warpPerspective(np.uint8(image), h1, (image.shape[1], image.shape[0]))
            # applies appearance augmentations to image1, the seed is fixed so that results are reproducible
            image1, image1_preprocessed = apply_augmentations(image1, self.cfg['augmentation'], seed=self.seed * (idx + 1))

            # sample homography and creates image2
            h2 = homography_sampling(image.shape, self.cfg['sample_homography'], seed=self.seed * (idx + 2))
            image2 = cv2.warpPerspective(np.uint8(image), h2, (image.shape[1], image.shape[0]))
            # applies appearance augmentations to image1
            image2, image2_preprocessed = apply_augmentations(image2, self.cfg['augmentation'], seed=self.seed * (idx + 2))

            # homography relatig image1 to image2
            H = np.matmul(h2, np.linalg.inv(h1))

            output = {'image1': torch.Tensor(image1.astype(np.int32)).unsqueeze(0), # put the images (gray) so that batch will be Bx1xHxW
                      'image2': torch.Tensor(image2.astype(np.int32)).unsqueeze(0),
                      'image1_normed': torch.Tensor(image1_preprocessed.astype(np.float32)).unsqueeze(0),
                      'image2_normed': torch.Tensor(image2_preprocessed.astype(np.float32)).unsqueeze(0),
                      'H1_to_2': torch.Tensor(H.astype(np.float32))}
            
        except:
            output = self.__getitem__(np.random.randint(0, idx - 1))

        return output

"""