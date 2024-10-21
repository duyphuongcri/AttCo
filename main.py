import os
import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn
import argparse 
from tqdm import tqdm 
from datetime import datetime 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import losses, transforms, dataset, metrics, utils
from torch.nn import init
# torch.backends.cudnn.enabled = True

def param_network(model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    #print(model)
    print("The number of parameters: {}".format(num_params))

def init_weights(net, init_type='xavier_uniform_', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier_normal_':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform_':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'kaiming_normal_':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform_':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1 or classname.find('GroupNorm') != -1:
            #init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    #print('Initialize network with %s' % init_type)
    net.apply(init_func)

if __name__=="__main__":
    date = (datetime.now()).strftime("%Y-%m-%d")
    frame = pd.read_csv("BraTS2020_Training_5folds.csv")
    for fold in range(5):
        print("Training fold: ", fold)
        parser = argparse.ArgumentParser()
        # hyper-parameters
        parser.add_argument('--epochs', type=int, default=200,
                            help='epoch number')
        parser.add_argument('--lrate', type=float, default=0.001,
                            help='learning rate')
        parser.add_argument('--train_batch_size', type=int, default=4,
                            help='training batch size')
        parser.add_argument('--val_batch_size', type=int, default=1,
                            help='validation batch size')
        parser.add_argument('--is_thop', type=bool, default=True,
                            help='whether calculate FLOPs/Params (Thop)')                     
        parser.add_argument('--path_image', type=str, default="/home/ssd_0/Phuong/MultimodalSegmentation/dataset/MICCAI_BraTS2020_TrainingData",
                            help='')
        parser.add_argument('--pretrained', type=str, default=None, 
                            help='')  
        parser.add_argument('--modelname', type=str, default="AttCo",
                            help='Choose one of models: ')    
        parser.add_argument('--dataname', type=str, default="BraTS2020",
                            help='Choose one of models: ')    

        arg = parser.parse_args()

        np.random.seed(100)
        torch.manual_seed(100)

        device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')


        if arg.modelname == "AttCo":
            import models.AttCo_BraTS as net
            model = net.AttCo(inChannel=2, outChannel=4, baseChannel=16)

        train_transforms = transforms.Compose([
                        transforms.NormalizeIntensity(),
                        transforms.RandomCrop(margin=(16,16,16), target_size=(128,128,128), original_size=(155,240,240)),
                        transforms.Mirroring(p=0.5),
                        transforms.ToTensor()])
        val_transforms = transforms.Compose([
                        transforms.NormalizeIntensity(),
                        transforms.RandomCrop(margin=(0,0,0), target_size=(128,128,128), original_size=(155,240,240)),
                        transforms.ToTensor()])

        listTrainPatients = list(frame["ID"][frame["Fold_{}".format(fold)]==1])
        listValPatients = list(frame["ID"][frame["Fold_{}".format(fold)]==0])
        train_set = dataset.MedDataset(arg.path_image, listTrainPatients,  transforms=train_transforms, mode="train")
        val_set = dataset.MedDataset(arg.path_image, listValPatients, transforms=val_transforms, mode="val")

        # Dataloaders:
        train_loader = DataLoader(train_set, batch_size=arg.train_batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=arg.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)

        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        ##############################################################################
        print(arg.modelname)
        # model = torch.nn.DataParallel(model)
        model.to(device)
        
        ### Load pretrained model###########################################
        if arg.pretrained is not None:
            print("Loading pretrained model")
            pretrain = torch.load(arg.pretrained)
            model_dict = model.state_dict()
            pretrained_dict = {k:v for k, v in pretrain.state_dict().items() if k in model.state_dict()}
            model_dict.update(pretrained_dict)
                
            model.load_state_dict(model_dict)
        else: 
            print("Training from Scratch")
            model.apply(init_weights)
        #######################################################################

        param_network(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lrate, betas=(0.9, 0.99))
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=0.0001)
        
        criterion = losses.CE_GeneralizedSoftDiceLoss() #nn.CrossEntropyLoss()
        dice_metric = metrics.DiceMetrics()

        min_loss = np.inf
        max_meanDice = 0
        ###################################
        filename = "model.pt"
        dice_max = 0
        log_data = []
        for epoch in range(arg.epochs):
            ### Trainingg phrase
            loss_train, loss_val = 0, 0
            dice_train, dice_val = [0]*4, [0]*4
            model.train()
            for sample in tqdm(dataloaders['train']):
                input, target = sample["input"].to(device), sample["target"].type(torch.LongTensor).to(device)
                output = model(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                # zero the parameter gradients:
                optimizer.zero_grad()

                dice = dice_metric(output, target)
                # Losses and metric:
                loss_train += loss.item()
                for idx_dice in range(4):
                    dice_train[idx_dice] += dice[idx_dice].item()

            loss_train /= len(dataloaders['train'])
            for idx_dice in range(4): 
                dice_train[idx_dice] /= len(dataloaders['train'])
            
            # Val phrase
            model.eval()
            with torch.no_grad():
                for sample in tqdm(dataloaders['val']):
                    input, target = sample["input"].to(device), sample["target"].type(torch.LongTensor).to(device)
                    #output = utils.post_processing(model, input)
                    output = model(input)
                    loss = criterion(output, target)
                    dice = dice_metric(output, target)

                    # Losses and metric:
                    loss_val += loss.item()
                    for idx_dice in range(4):
                        dice_val[idx_dice] += dice[idx_dice].item()

            loss_val /= len(dataloaders['val'])
            for idx_dice in range(4): 
                dice_val[idx_dice] /= len(dataloaders['val']) 
            
            # Adjust learning rate after val phase:
            # scheduler.step()
            
            # Save model
            # torch.save(model, "model_last_epoch.pt")
            if min_loss > loss_val:
                min_loss = loss_val
                if os.path.exists(filename):
                    os.remove(filename)    
                
                filename = "./checkpoint/{}/{}/Fold_{}_bs_{}_TC_{}_ED_{}_ET_{}_WT_{}.pt".format(arg.dataname, arg.modelname, fold, arg.train_batch_size, dice_val[0], dice_val[1], dice_val[2], dice_val[3])
                torch.save(model, filename)
                print("Saving model: ", filename)
            print("Epoch: {} | Loss_train: {:.04f} | Dice_TC: {:.04f} | Dice_ED: {:.04f} | Dice_ET {:.04f} | Dice_WT: {:.04f}".format(epoch, loss_train, dice_train[0], dice_train[1], dice_train[2], dice_train[3]))
            print("Epoch: {} | Loss_val: {:.04f} | Dice_TC: {:.04f} | Dice_ED: {:.04f} | Dice_ET {:.04f} | Dice_WT: {:.04f}".format(epoch, loss_val, dice_val[0], dice_val[1], dice_val[2], dice_val[3]))
            log_data.append([epoch, loss_train, dice_train[0], dice_train[1], dice_train[2], dice_train[3], loss_val, dice_val[0], dice_val[1], dice_val[2], dice_val[3]])

            # Save log
            log_data_frame = np.asarray(log_data)
            log_data_frame = pd.DataFrame(log_data_frame, columns=[['Epoch', 'Loss_train', 'Dice_TC_train', 'Dice_ED_train', 'Dice_ET_train', 'Dice_WT_train', 'Loss_val', 'Dice_TC_val', 'Dice_ED_val', 'Dice_ET_val', 'Dice_WT_val']])
            log_data_frame.to_csv("./checkpoint/{}/{}/log_train_BraTS2020_fold_{}_bs_{}.csv".format(arg.dataname, arg.modelname, fold, arg.train_batch_size), index=False)
                

            
