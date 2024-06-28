import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

def check_and_reshape(img):
    if len(img.shape) == 2: 
        img = img.unsqueeze(0)  
    return img

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('/system/user/studentwork/hludwig/glam/GLAMpoints_pytorch-master/training_validation_loss.png')
    plt.show()

def plot_training(image1, image2, kp_map1, kp_map2, computed_reward1, loss, mask_batch1, metrics_per_image,
                  epoch, save_path, name_to_save):

    fig, ((axis1, axis2), (axis3, axis4), (axis5, axis6), (axis7, axis8)) = \
        plt.subplots(4, 2, figsize=(20, 20))
    nbr=0
    size_image = image1[nbr, :, :].shape
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0002, hspace=0.4)
    axis1.imshow(image1[nbr, :, :], cmap='gray', vmin=0, vmax=255)
    axis1.scatter(metrics_per_image['{}'.format(nbr)]['to_plot']['tp_kp1'][1],
                  metrics_per_image['{}'.format(nbr)]['to_plot']['tp_kp1'][0], s=4, color='red')
    axis1.set_title('epoch{}, img1:original_image, \n true positive points in red'.format(epoch), fontsize='small')

    im = axis2.imshow(image2[nbr, :, :], cmap='gray', vmin=0, vmax=255)
    axis2.scatter(metrics_per_image['{}'.format(nbr)]['to_plot']['warped_tp_kp1'][1],
                  metrics_per_image['{}'.format(nbr)]['to_plot']['warped_tp_kp1'][0], s=2,
                  color='red')
    axis2.set_title('img2:warped_image', fontsize='small')
    fig.colorbar(im, ax=axis2)
    axis3.imshow(kp_map1[nbr, :, :], vmin=0.0, vmax=1.0, interpolation='nearest')
    axis3.set_title('kp map of image1, max {}, min {}'.format(np.max(kp_map1[nbr]),
                                                              np.min(kp_map1[nbr])),
                    fontsize='small')
    im4 = axis4.imshow(kp_map2[nbr, :, :], vmin=0.0, vmax=1.0, interpolation='nearest')
    axis4.set_title('kp map of image2', fontsize='small')
    fig.colorbar(im4, ax=axis4)
    axis5.imshow(image1[nbr, :, :], cmap='gray', origin='upper', vmin=0.0, vmax=255.0)
    axis5.scatter(metrics_per_image['{}'.format(nbr)]['to_plot']['keypoints_map1'][1],
                  metrics_per_image['{}'.format(nbr)]['to_plot']['keypoints_map1'][0], s=2,
                  color='green')
    axis5.scatter(metrics_per_image['{}'.format(nbr)]['to_plot']['tp_kp1'][1],
                  metrics_per_image['{}'.format(nbr)]['to_plot']['tp_kp1'][0], s=4, color='red')
    axis5.set_title('kp_map after NMS of image1 in green, nbr_kp {}, \n nbr tp keypoints found {} (in red)'.format(
        metrics_per_image['{}'.format(nbr)]['nbr_kp1'],
        metrics_per_image['{}'.format(nbr)]['total_nbr_kp_reward1']), fontsize='medium')

    im6 = axis6.imshow(image2[nbr, :, :], cmap='gray', origin='upper', vmin=0.0, vmax=255.0)
    axis6.scatter(metrics_per_image['{}'.format(nbr)]['to_plot']['keypoints_map2'][1],
                  metrics_per_image['{}'.format(nbr)]['to_plot']['keypoints_map2'][0], s=2,
                  color='green')
    axis6.scatter(metrics_per_image['{}'.format(nbr)]['to_plot']['warped_tp_kp1'][1],
                  metrics_per_image['{}'.format(nbr)]['to_plot']['warped_tp_kp1'][0], s=4,
                  color='red')
    axis6.set_title('kp_map after NMS of image2 in green, nbr_kp {} \n true positive points in red'.format(
        metrics_per_image['{}'.format(nbr)]['nbr_kp2']), fontsize='small')
    fig.colorbar(im6, ax=axis6)

    axis7.imshow(computed_reward1[nbr, :, :], vmin=0, vmax=1, interpolation='nearest')
    axis7.set_title('computed reward:,\nrepeatability={}, total_nbr_tp_kp={} \n binary mask for backpropagation sum={}'.format(
        metrics_per_image['{}'.format(nbr)]['repeatability'],
        metrics_per_image['{}'.format(nbr)]['total_nbr_kp_reward1'],
    np.sum(mask_batch1[nbr, :, :])), fontsize='small')

    im8 = axis8.imshow(
        cv2.warpPerspective(image1[nbr, :, :], metrics_per_image['{}'.format(nbr)]
        ['computed_H'], (size_image[1], size_image[0])), cmap='gray', vmin=0, vmax=255)
    axis8.set_title('warped image 1 according to computed homography,\n inlier_ratio={}, '
                     'true_homo={}, class_acceptable={}'.format(
        metrics_per_image['{}'.format(nbr)]['inlier_ratio'],
        metrics_per_image['{}'.format(nbr)]['homography_correct'],
        metrics_per_image['{}'.format(nbr)]['class_acceptable']), fontsize='small')
    fig.colorbar(im8, ax=axis8)
    fig.savefig(Path(save_path) / (name_to_save), bbox_inches='tight')
    plt.close(fig)

