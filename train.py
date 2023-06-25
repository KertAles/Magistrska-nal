# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:32:12 2023

@author: Kert PC
"""

from torch.utils.data import DataLoader
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.std import tqdm
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
from torch.optim.lr_scheduler import MultiStepLR

from models import UNet, UNetDouble
from dataset_grid import SatDataset
from loss_log_likelihood import LogLikelihoodLoss

TRAIN_DATA_PATH = './data/npy/train/'
VAL_DATA_PATH = './data/npy/val/'
MODEL_PATH = './models/'
T_WINDOW = 27

EPOCHS = 60

BATCH_SIZE = 6

def train(net, trainset, trainloader, valset, valloader, out_channels, epochs, checkpoint_count=2) :

    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.25)

    log_likelihood_loss = LogLikelihoodLoss()
    gnn_loss = torch.nn.GaussianNLLLoss()
    mse_loss = torch.nn.MSELoss(reduction='sum')
    
    f_t = open(f"loss{out_channels}_train.txt", "w")
    f_v = open(f"loss{out_channels}_val.txt", "w")
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        with tqdm(total=len(trainset), desc ='Epoch: '+str(epoch)+"/"+str(epochs), unit='img') as prog_bar:
            for i, data in enumerate(trainloader, 0):
                  # get the inputs;
                  sat_data = data['satData']
                  gt = data['gt']
                  mask = data['mask']
    
                  # Get the inputs to the GPU
                  sat_data = sat_data.cuda().float()
                  gt = gt.cuda().float()
                  mask = mask.cuda().bool()
    
                  # zero the parameter gradients
                  optimizer.zero_grad()
    
                  # forward + backward + optimize
                  outputs = net(sat_data)
                  #loss = log_likelihood_loss(gt, outputs, mask)
                  
                  if out_channels == 1 :
                      outputs_masked = outputs[:, 0, :, :] * mask
                      loss = mse_loss(outputs_masked, gt) / torch.sum(mask)
                  else :
                      pred1 = outputs[:, 0, :, :]
                      pred2 = outputs[:, 1, :, :]
                      
                      var, mean = log_likelihood_loss.get_var_and_mean(pred1, pred2)
                      
                      mean_masked = mean * mask
                      loss = mse_loss(mean_masked, gt) / torch.sum(mask)
                  
                  #loss = gnn_loss(mean, gt, var)
                  loss.backward()
                  optimizer.step()
                  running_loss += loss.data.cpu().detach().numpy()
    
                  prog_bar.set_postfix(**{'loss': loss.data.cpu().detach().numpy()})
                  prog_bar.update(BATCH_SIZE)
                  
            if epoch % checkpoint_count == 0 :
                torch.save(net.state_dict(), MODEL_PATH + f'checkpoint_ch{out_channels}_{epoch}')
            running_loss /= (i+1) 
            f_t.write(f'{running_loss}\n')
            scheduler.step()
            
            prog_bar.set_postfix(**{'loss': running_loss})
            
            running_loss = 0.0
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(valloader, 0):
                          # get the inputs;
                          sat_data = data['satData']
                          gt = data['gt']
                          mask = data['mask']
            
                          # Get the inputs to the GPU
                          sat_data = sat_data.cuda().float()
                          gt = gt.cuda().float()
                          mask = mask.cuda().bool()
        
                          outputs = net(sat_data)
                          
                          if out_channels == 1 :
                              outputs_masked = outputs[:, 0, :, :] * mask
                              loss = mse_loss(outputs_masked, gt) / torch.sum(mask)
                          else :
                              pred1 = outputs[:, 0, :, :]
                              pred2 = outputs[:, 1, :, :]
                              
                              var, mean = log_likelihood_loss.get_var_and_mean(pred1, pred2)
                              
                              mean_masked = mean * mask
                              loss = mse_loss(mean_masked, gt) / torch.sum(mask)
                              
                          RMS = torch.sqrt(loss)
                          
                          running_loss += RMS.data.cpu().detach().numpy()
                          
            running_loss /= (i+1)
            f_v.write(f'{running_loss}\n')
            net.train()
            print(f'Validation RMS at epoch {epoch} : {running_loss}')
            
    f_t.close()
    f_v.close()
    
    return net


def train_double(net, trainset, trainloader, valset, valloader, out_channels, epochs, weights=[0.25, 0.75], checkpoint_count=2) :
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.25)

    log_likelihood_loss = LogLikelihoodLoss()
    gnn_loss = torch.nn.GaussianNLLLoss()
    mse_loss = torch.nn.MSELoss(reduction='sum')
    
    f_t = open(f"loss{out_channels}_double_train.txt", "w")
    f_v = open(f"loss{out_channels}_double_val.txt", "w")
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        with tqdm(total=len(trainset), desc ='Epoch: '+str(epoch)+"/"+str(epochs), unit='img') as prog_bar:
            for i, data in enumerate(trainloader, 0):
                  # get the inputs;
                  sat_data = data['satData']
                  gt = data['gt']
                  mask = data['mask']
    
                  # Get the inputs to the GPU
                  sat_data = sat_data.cuda().float()
                  gt = gt.cuda().float()
                  mask = mask.cuda().bool()
    
                  # zero the parameter gradients
                  optimizer.zero_grad()
    
                  # forward + backward + optimize
                  output_end, output_mid = net(sat_data)
                  #loss = log_likelihood_loss(gt, outputs, mask)
                  
                  if out_channels == 1 :
                      output_mid_masked = output_mid[:, 0, :, :] * mask
                      loss_mid = mse_loss(output_mid_masked, gt)
                      
                      output_end_masked = output_end[:, 0, :, :] * mask
                      loss_end = mse_loss(output_end_masked, gt)
                      
                  else :
                      pred1_mid = output_mid[:, 0, :, :]
                      pred2_mid = output_mid[:, 1, :, :]
                      
                      var, mean_mid = log_likelihood_loss.get_var_and_mean(pred1_mid, pred2_mid)
                      mean_mid_masked = mean_mid * mask
                      loss_mid = mse_loss(mean_mid_masked, gt)
                      
                      pred1_end = output_end[:, 0, :, :]
                      pred2_end = output_end[:, 1, :, :]
                      
                      var, mean_end = log_likelihood_loss.get_var_and_mean(pred1_end, pred2_end)
                      mean_end_masked = mean_end * mask
                      loss_end = mse_loss(mean_end_masked, gt) 
                      
                  loss = (loss_mid * weights[0] + loss_end * weights[1]) / torch.sum(mask)
                  
                  #loss = gnn_loss(mean, gt, var)
                  loss.backward()
                  optimizer.step()
                  running_loss += loss.data.cpu().detach().numpy()
    
                  prog_bar.set_postfix(**{'loss': loss.data.cpu().detach().numpy()})
                  prog_bar.update(BATCH_SIZE)
                  
            if epoch % checkpoint_count == 0 :
                torch.save(net.state_dict(), MODEL_PATH + f'checkpoint_ch{out_channels}_double_{epoch}')
            running_loss /= (i+1) 
            f_t.write(f'{running_loss}\n')
            scheduler.step()
            
            prog_bar.set_postfix(**{'loss': running_loss})
            
            running_loss = 0.0
            net.eval()
            
        
            with torch.no_grad():
                    for i, data in enumerate(valloader, 0):
                          # get the inputs;
                          sat_data = data['satData']
                          gt = data['gt']
                          mask = data['mask']
            
                          # Get the inputs to the GPU
                          sat_data = sat_data.cuda().float()
                          gt = gt.cuda().float()
                          mask = mask.cuda().bool()
        
                          output_end, output_mid = net(sat_data)
                          
                          if out_channels == 1 :
                              output_end_masked = output_end[:, 0, :, :] * mask
                              loss_end = mse_loss(output_end_masked, gt)
                          else :
                              pred1_end = output_end[:, 0, :, :]
                              pred2_end = output_end[:, 1, :, :]
                              
                              var, mean_end = log_likelihood_loss.get_var_and_mean(pred1_end, pred2_end)
                              mean_end_masked = mean_end * mask
                              loss_end = mse_loss(mean_end_masked, gt) 
                              
                          RMS = torch.sqrt(loss_end / torch.sum(mask))
                          
                          running_loss += RMS.data.cpu().detach().numpy()
                          
            running_loss /= (i+1)
            f_v.write(f'{running_loss}\n')
            net.train()
            print(f'Validation RMS at epoch {epoch} : {running_loss}')
    f_t.close()
    f_v.close()
    
    return net



if __name__ == "__main__":
    
    trainset = SatDataset(TRAIN_DATA_PATH, T_WINDOW)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=1)
    
    
    valset = SatDataset(VAL_DATA_PATH, T_WINDOW)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=1)
    
    in_channels = T_WINDOW + 4
    train_double_net = True
    out_channels_list = [1, 2]
    for out_channels in out_channels_list :
        #if train_double_net :
            net = UNet(in_channels=in_channels, out_channels=out_channels, depths=[32, 64, 128, 256]).cuda().float()
            summary(net, (in_channels, 176, 64), device='cuda')
            
            net = train(net, trainset, trainloader, valset, valloader, out_channels, EPOCHS) 
            torch.save(net.state_dict(), MODEL_PATH + f'model_ch{out_channels}_final')
            
            net = UNetDouble(in_channels=in_channels, out_channels=out_channels, depths=[32, 64, 128, 256]).cuda().float()
            summary(net, (in_channels, 176, 64), device='cuda')
            
            net = train_double(net, trainset, trainloader, valset, valloader, out_channels, EPOCHS) 
            torch.save(net.state_dict(), MODEL_PATH + f'model_ch{out_channels}_double_final')
        #else :
            
        
    #display_preds(net_skips, net_noskips, testset)
    