import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class CE_GeneralizedSoftDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CE_GeneralizedSoftDiceLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.diceloss = GeneralizedSoftDiceLoss()
        self.alpha=alpha
    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, D, H, W)
        args: label: tensor of shape(N, D, H, W)
        '''
        # print(logits.shape, label.shape)
        loss = self.ce_loss(logits, label)*self.alpha + self.diceloss(logits, label)*(1-self.alpha)
        return loss

class CE_GeneralizedSoftDiceLoss_v2(nn.Module):
    def __init__(self, alpha=0.5):
        super(CE_GeneralizedSoftDiceLoss_v2, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.alpha=alpha

    def diceloss(self, probs, labels):
        N = len(probs)
        smooth = 1.0    
        m1  = probs.view (N, -1)
        m2  = labels.view(N, -1)
        intersection = (m1 * m2).sum(dim=1)
        score = (2.0 * intersection + smooth) / (m1.sum(dim=1) + m2.sum(dim=1) + smooth)
        return (1- score).sum() /N 
    
    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, D, H, W)
        args: label: tensor of shape(N, D, H, W)
        '''
        # print(logits.shape, label.shape)
        prob = torch.softmax(logits, dim=1)[:, 1:]
        loss =  self.diceloss(prob, label)*(1-self.alpha)

        label = label.squeeze(1).type(torch.LongTensor).to(logits.device)
        
        loss += self.ce_loss(logits, label)*self.alpha 
        return loss
    
class BCE_SoftDiceLoss_v1(nn.Module):
    def __init__(self, alpha=0.5):
        super(BCE_SoftDiceLoss_v1, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.diceloss = SoftDiceLoss_v1()
        self.alpha=alpha
    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, D, H, W)
        args: label: tensor of shape(N, D, H, W)
        '''
        loss = self.bce_loss(logits, label) + self.diceloss(logits, label)
        return loss

class GeneralizedSoftDiceLoss(nn.Module):
    def __init__(self,
                 p=1,
                 smooth=0.0001,
                 reduction='mean',
                 weight=True,
                 ignore_lb=255):
        super(GeneralizedSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction
        self.weight = weight
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, D, H, W)
        args: label: tensor of shape(N, D, H, W)
        '''
        # overcome ignored label
        logits = logits.float()
        ignore = label.data.cpu() == self.ignore_lb
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0
        lb_one_hot = lb_one_hot.detach()
        
        # compute loss
        probs = torch.softmax(logits, dim=1)

        numer = torch.sum((probs*lb_one_hot), dim=(2, 3, 4))
        denom = torch.sum(probs.pow(self.p)+lb_one_hot.pow(self.p), dim=(2, 3, 4))
        if self.weight:
            weight = lb_one_hot.sum((2, 3,4))
            weight = 1 / (weight).clamp(min=self.smooth)
            # print(weight)
            weight.requires_grad = False

            numer = numer * weight
            denom = denom * weight

        numer = torch.sum(numer[:, 1:], dim=1)
        denom = torch.sum(denom[:, 1:], dim=1)
        loss = 1 - (2*numer+self.smooth)/(denom+self.smooth)

        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()
    def forward(self, logits, labels):
        N = len(logits)
        logits = nn.Sigmoid()(logits)
        
        logits_flat = logits.view(N, -1)
        labels_flat = labels.view(N, -1)
        return self.bce_loss(logits_flat, labels_flat)

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
    def forward(self, y_true, y_pred):
        loss = (abs(y_true - y_pred)).sum() / y_true.shape[0]
        return loss  

class SoftDiceLoss_v1(nn.Module):
    def __init__(self):  
        super(SoftDiceLoss_v1, self).__init__()

    def forward(self, logits, labels):
        N = len(logits)
        probs = torch.sigmoid(logits)
        smooth = 1.0    
        m1  = probs.view (N, -1)
        m2  = labels.view(N, -1)
        intersection = (m1 * m2).sum(dim=1)
        score = (2.0 * intersection + smooth) / (m1.sum(dim=1) + m2.sum(dim=1) + smooth)
        return (1- score).sum() /N 


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        N = len(inputs)
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(N, -1)
        targets = targets.view(N, -1)
        
        intersection = (inputs * targets).sum(dim=1)                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum(dim=1) + targets.sum(dim=1) + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss.sum()/N
        
        return Dice_BCE

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU


