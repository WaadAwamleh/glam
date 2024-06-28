import numpy as np
from tqdm import tqdm
import pickle
import random
import json
from datetime import datetime
from pathlib import Path
from termcolor import colored
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from utils.dataset import SyntheticDataset
from utils.utils import check_and_reshape, plot_losses, plot_training
from utils.loss import compute_reward, compute_loss
from utils.logger import LoggerUtility
from utils.checkpoint_manager import get_most_recent_checkpoint,load_checkpoint, save_checkpoint
from utils.unet import UNet
#from utils.glam import UNet



def main(con: dict, args) -> None:

    n_jobs = con['n_jobs']
    dataset = con['dataset']
    batch_size = con['batch_size']
    lr = con['learning_rate']
    weight_decay = con['weight_decay']
    epochs = con['epochs']
    verbose = con['compute_metrics']
    
    logger = LoggerUtility.get_logger()
    cwd = Path.cwd()
    model_path = cwd.joinpath('model')
    if not model_path.exists():
        model_path.mkdir(mode=0o770, parents=True, exist_ok=True)
    
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    date = datetime.now()
    name = date.strftime('%Y%m%d')
    train_losses = []
    val_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = SyntheticDataset(Path(dataset) , train=True)
    val_dataset = SyntheticDataset(Path(dataset) , train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_jobs)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=n_jobs)
    model = UNet(attention=False)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=lr,weight_decay=weight_decay)
    save_path = Path(con['write_dir']) / con['name_exp']
    if con["load_pretrained_weights"]:
        checkpoints = Path(con["pretrained_weights"])
        if not checkpoints.exists():
            logger.error('The path to the model pre-trained weights you indicated does not exist.')
        else:
            filename = get_most_recent_checkpoint(checkpoints)
            logger.info('path to pretrained model is {}'.format(checkpoints))      
            model, optimizer, start_epoch, best_val = load_checkpoint(model, optimizer, filename=filename)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)       
    else:
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / 'name.pkl', 'wb') as f: #change it to f'{name}.pkl'
            pickle.dump(name, f)  
        best_val = -1
        start_epoch = 0   
    model = nn.DataParallel(model)
    model = model.to(device)
    save_train = Path(save_path) / 'train'
    save_train .mkdir(parents=True,exist_ok=True)
    save_valid = Path(save_path) / 'valid'
    save_valid .mkdir(parents=True,exist_ok=True)
    for epoch in range(start_epoch, epochs):
        print('starting epoch {}'.format(epoch))
        if epoch == 0:
            nms = con['nms0']
        else:
            nms = con['nms']   
        n_iter = epoch*len(train_dataloader)
        model.train()
        running_train_loss = 0
        if verbose:
            nbr_images = 0.0
            nbr_homo_correct = 0.0
            nbr_homo_accept = 0.0
            nbr_tp = 0.0
            nbr_kp = 0.0
            add_repeatability = 0.0
            
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, mini_batch in pbar:
            image1 = check_and_reshape(mini_batch['image1'].to(device))
            image2 = check_and_reshape(mini_batch['image2'].to(device))
            image1_normed = mini_batch['image1_normed'].to(device)
            image2_normed = mini_batch['image2_normed'].to(device)
            homographies = mini_batch['H1_to_2']
            optimizer.zero_grad()           
            kp_map1 = model(check_and_reshape(image1_normed)) # shape Bx1xHxW
            kp_map2 = model(check_and_reshape(image2_normed))
            computed_reward1, mask_batch1, metrics_per_image = compute_reward(image1, 
                                                                              image2, 
                                                                              kp_map1.clone().detach(),
                                                                              kp_map2.clone().detach(),
                                                                              homographies, 
                                                                              nms,
                                                                              distance_threshold=con['distance_threshold'],
                                                                              device=device,
                                                                              compute_metrics=verbose)
            
            if verbose:
                nbr_images += metrics_per_image['nbr_images']
                nbr_homo_correct += metrics_per_image['nbr_homo_correct']
                nbr_homo_accept += metrics_per_image['nbr_homo_acceptable']
                nbr_tp += metrics_per_image['nbr_tp']
                nbr_kp += metrics_per_image['nbr_kp']
                add_repeatability += metrics_per_image['sum_rep']
                logger.info(f'cumulative_ratio_correct_homographies_per_iter, {float(nbr_homo_correct) / nbr_images}, {n_iter}')
                logger.info(f'cumulative_ratio_acceptable_homographies_per_iter {float(nbr_homo_accept) / nbr_images}, {n_iter}')
                logger.info(f'cumulative_average_number_of_kp_per_iter {float(nbr_kp) / nbr_images}, {n_iter}')
                logger.info(f'cumulative_average_number_of_tp_per_iter {float(nbr_tp) / nbr_images}, {n_iter}')
                logger.info(f'cumulative_repeatability_per_iter, {float(add_repeatability) / nbr_images}, {n_iter}')
            Loss = compute_loss(reward=computed_reward1, kpmap=kp_map1, mask=mask_batch1)
            Loss.backward()
            optimizer.step()
            
            if i % con["plot_every_x_batches"] == 0 and verbose:
                plot_training(image1.cpu().numpy().squeeze(1), image2.cpu().numpy().squeeze(1),
                            kp_map1.detach().cpu().numpy().squeeze(1), kp_map2.detach().cpu().numpy().squeeze(1),
                            computed_reward1.cpu().numpy().squeeze(1), Loss.item(),
                            mask_batch1.cpu().numpy().squeeze(1), metrics_per_image, epoch, save_train,
                            name_to_save='epoch{}_batch{}.jpg'.format(epoch, i))
            running_train_loss += Loss.item()
            logger.info('train_loss_per_iter', Loss.item(), n_iter)
            n_iter += 1
            pbar.set_description('training: R_total_loss: %.3f/%.3f' % (running_train_loss / (i + 1),Loss.item()))
            
        running_train_loss /= len(train_dataloader)
        train_losses.append(running_train_loss)
        logger.info('train loss', running_train_loss, epoch)
        print(colored('==> ', 'green') + 'Train average loss:', running_train_loss)
        if con['validation']:
            n_iter = epoch * len(val_dataloader)
            model.eval()
            running_valid_loss = 0
            if verbose:
                nbr_images = 0.0
                nbr_homo_correct = 0.0
                nbr_homo_accept = 0.0
                nbr_tp = 0.0
                nbr_kp = 0.0
                add_repeatability = 0.0
            with torch.no_grad():
                pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
                for i, mini_batch in pbar:
                    image1 = mini_batch['image1'].to(device)
                    image2 = mini_batch['image2'].to(device)
                    image1_normed = mini_batch['image1_normed'].to(device)
                    image2_normed = mini_batch['image2_normed'].to(device)
                    homographies = mini_batch['H1_to_2']
                    kp_map2 = model(image2_normed)
                    kp_map1 = model(image1_normed)
                    computed_reward1, mask_batch1, metrics_per_image = compute_reward(image1,
                                                                                      image2,
                                                                                      kp_map1.clone(),
                                                                                      kp_map2.clone(),
                                                                                      homographies, 
                                                                                      nms,
                                                                                      distance_threshold=con['distance_threshold'],
                                                                                      device=device,
                                                                                      compute_metrics=verbose)
                    if verbose:
                        nbr_images += metrics_per_image['nbr_images']
                        nbr_homo_correct += metrics_per_image['nbr_homo_correct']
                        nbr_homo_accept += metrics_per_image['nbr_homo_acceptable']
                        nbr_tp += metrics_per_image['nbr_tp']
                        nbr_kp += metrics_per_image['nbr_kp']
                        add_repeatability += metrics_per_image['sum_rep']
                        logger.info(f'cumulative_ratio_correct_homographies_per_iter, {float(nbr_homo_correct)/nbr_images}, {n_iter}')
                        logger.info(f'cumulative_ratio_acceptable_homographies_per_iter, {float(nbr_homo_accept)/nbr_images}, {n_iter}')
                        logger.info(f'cumulative_average_number_of_kp_per_iter, {float(nbr_kp)/nbr_images}, {n_iter}')
                        logger.info(f'cumulative_average_number_of_tp_per_iter, {float(nbr_tp)/nbr_images}, {n_iter}')
                        logger.info(f'cumulative_repeatability_per_iter, {float(add_repeatability)/nbr_images}, {n_iter}')
                    Loss = compute_loss(reward=computed_reward1, kpmap=kp_map1, mask=mask_batch1)
                    
                    if i < 2 and verbose:
                        plot_training(image1.cpu().numpy().squeeze(), image2.cpu().numpy().squeeze(),
                                    kp_map1.cpu().numpy().squeeze(), kp_map2.cpu().numpy().squeeze(),
                                    computed_reward1.cpu().numpy().squeeze(), 
                                    Loss.item(),
                                    mask_batch1.cpu().numpy().squeeze(), 
                                    metrics_per_image, epoch, save_valid,
                                    name_to_save='epoch{}_batch{}'.format(epoch, i))
                    running_valid_loss += Loss.item()
                    logger.info('val_loss_per_iter', Loss.item(), n_iter)
                    n_iter += 1
                    pbar.set_description('validation: R_total_loss: %.3f/%.3f' % (running_valid_loss / (i + 1),Loss.item()))
                running_valid_loss /= len(val_dataloader)
                
                if verbose:
                    logger.info('ratio_correct_homographies_per_epoch',float(nbr_homo_correct) / nbr_images, epoch)
                    logger.info('ratio_acceptable_homographies_per_epoch',float(nbr_homo_accept) / nbr_images, epoch)
                    logger.info('average_number_of_kp_per_epoch', float(nbr_kp) / nbr_images, epoch)
                    logger.info('average_number_of_tp_per_epoch', float(nbr_tp) / nbr_images, epoch)
                    logger.info('repeatability_per_epoch', float(add_repeatability) / nbr_images, epoch)
            val_loss =  running_valid_loss / len(val_dataloader)
            val_losses.append(val_loss)
            print(colored('==> ', 'blue') + 'Val average grid loss :', val_loss)
            print(colored('==> ', 'blue') + 'epoch :', epoch + 1)
            logger.info('val loss', val_loss, epoch)
            if best_val < 0:
                best_val = val_loss
            is_best = val_loss < best_val
            best_val = min(val_loss, best_val)
        else:
            is_best = False
        save_checkpoint({'epoch': epoch + 1,'state_dict': model.module.state_dict(),'optimizer': optimizer.state_dict(),
                         'best_loss': best_val},is_best, save_valid, 'epoch_{}.pth'.format(epoch + 1))
    plot_losses(train_losses, val_losses)
if __name__ == '__main__':
    from argparse import ArgumentParser
    import json
    local_config = Path(__file__).resolve().parent / "configs"
    local_config.mkdir(parents=True, exist_ok=True)
    local_config = Path(local_config / "train.json")
    parser = ArgumentParser(description="process and configure dataset handling")
    parser.add_argument("--data_path",dest="data_path",default=None,help="Path to overwrite data input from config file")
    args = parser.parse_args()
    LoggerUtility.setup_logging(local_config)
    with open(local_config, "r", encoding='utf-8') as config_file:
        config = json.load(config_file)
    if args.data_path:
        config['dataset'] = args.data_path
    main(config, args)