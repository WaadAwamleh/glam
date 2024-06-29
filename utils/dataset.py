import random
import cv2
import numpy as np
import torch
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
        self.list_original_images = [f for f in self.root_dir.rglob('*healthy/*.JPG')] #*healthy/
        if train:
            self.list_original_images = self.list_original_images[:int(0.8 * len(self.list_original_images))]
        else:
            self.list_original_images = self.list_original_images[int(0.8 * len(self.list_original_images)):]

    def __len__(self):
        return len(self.list_original_images)

    def __getitem__(self, idx):
        random.seed(23)
        name_image = self.list_original_images[idx]
        
        #path_original_image = Path(self.root_dir) / str(name_image)
        image_full = cv2.imread(str(name_image))
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
