import torch 
import torch.nn as nn
from scipy import ndimage
import numpy as np 

class DiceMetrics(nn.Module): ### for BraTS dataset
    def __init__(self):  #weight=None, size_average=True):
        super(DiceMetrics, self).__init__()
        self.smooth = 1e-12

    def dice(self, probs, labels):
        N = len(probs)
        m1  = probs.view (N, -1)
        m2  = labels.view(N, -1)
        
        intersection = (m1 * m2).sum(dim=1)
        score = (2.0 * intersection) / (m1.sum(dim=1) + m2.sum(dim=1) + self.smooth)
        return score.mean()

    def forward(self, logits, labels):
        """
        logits: B x C x D x H x W
        labels: B x D x H x W
        """
        C = logits.shape[1]
        probs = torch.argmax(logits, dim=1)
        # print(torch.unique(probs), torch.unique(labels))
        probs = torch.eye(C).to(probs.device)[probs] # B x D x H x W x C
        labels = torch.eye(C).to(probs.device)[labels] # B x D x H x W x C
        # dice_NCR = self.dice(probs[:,:,:,:,1], labels[:,:,:,:,1])
        dice_ED = self.dice(probs[:,:,:,:,2], labels[:,:,:,:,2])
        dice_ET = self.dice(probs[:,:,:,:,3], labels[:,:,:,:,3])
        dice_TC = self.dice(probs[:,:,:,:,1]+ probs[:,:,:,:,3], labels[:,:,:,:,1]+labels[:,:,:,:,3])
        dice_WT = self.dice(probs[:,:,:,:,1:].sum(-1), labels[:,:,:,:,1:].sum(-1))
        # print(dice_TC, dice_ED, dice_ET, dice_WT)
        return dice_TC, dice_ED, dice_ET, dice_WT
    
class DiceMetric_v2(nn.Module): ### for BraTS dataset
    def __init__(self):  #weight=None, size_average=True):
        super(DiceMetric_v2, self).__init__()
        self.smooth = 1e-12

    def dice(self, probs, labels):
        N = len(probs)
        m1  = probs.view (N, -1)
        m2  = labels.view(N, -1)
        
        intersection = (m1 * m2).sum(dim=1)
        score = (2.0 * intersection) / (m1.sum(dim=1) + m2.sum(dim=1) + self.smooth)
        return score.mean()

    def forward(self, logits, labels):
        """
        logits: B x C x D x H x W
        labels: B x D x H x W
        """
        C = logits.shape[1]
        probs = torch.argmax(logits, dim=1)
        probs = torch.eye(C).to(probs.device)[probs] # B x D x H x W x C
        labels = labels.squeeze(1).type(torch.LongTensor).to(logits.device)
        labels = torch.eye(C).to(probs.device)[labels] # B x D x H x W x C
        dice = self.dice(probs, labels)
  
        return dice

class DiceAccuracy_v1(nn.Module):
    def __init__(self, prob_mode=False):  #weight=None, size_average=True):
        super(DiceAccuracy_v1, self).__init__()
        self.prob_mode = prob_mode

    def forward(self, logits, labels):
        N = len(logits)
        if self.prob_mode:
            probs = logits
        else:
            probs = torch.sigmoid(logits)
        probs = (probs > 0.5).float()
        smooth = 1e-12
             
        m1  = probs.view (N, -1)
        m2  = labels.view(N, -1)
        
        intersection = (m1 * m2).sum(dim=1)
        score = (2.0 * intersection) / (m1.sum(dim=1) + m2.sum(dim=1) + smooth)
        
        return score.sum()/N

def numeric_score(logits, labels):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""
    probs = torch.sigmoid(logits)
    probs = (probs > 0.5).float()

    FP = ((probs == 0) & (labels == 1)).sum(dim=1)
    FN = ((probs == 1) & (labels == 0)).sum(dim=1)
    TP = ((probs == 1) & (labels == 1)).sum(dim=1)
    TN = ((probs == 0) & (labels == 0)).sum(dim=1)   

    return FP, FN, TP, TN


class PrecisionandRecall(nn.Module):
    def __init__(self):
        super(PrecisionandRecall, self).__init__()

    def forward(self, logits, labels):
        N = len(logits)
        logits = logits.view(N, -1)
        labels = labels.view(N, -1)

        FP, FN, TP, TN = numeric_score(logits, labels)

        precision = TP/(TP+FP+1e-12)
        recall = TP/(TP+FN+1e-12) 
        return precision.sum()/N, recall.sum()/N
    
    
class Recall(nn.Module):
    def __init__(self):
        super(Recall, self).__init__()

    def forward(self, logits, labels):
        N = len(logits)
        logits = logits.view(N, -1)
        labels = labels.view(N, -1)

        FP, FN, TP, TN = numeric_score(logits, labels)
        recall = TP/(TP+FN+1e-12)  

        return recall.sum()/N

class IouAccuracy_v1(nn.Module):
    def __init__(self):  #weight=None, size_average=True):
        super(IouAccuracy_v1, self).__init__()

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        probs = (probs > 0.5).float()
        smooth = 1e-12

        N = len(logits)

        m1  = probs.view (N, -1)
        m2  = labels.view(N, -1)
        
        intersection = (m1 * m2).sum(dim=1)
        union = m1.sum(dim=1) + m2.sum(dim=1) - (m1*m2).sum(dim=1)
        score = intersection / (union + smooth)
        
        return score.sum()/N

def border_map(binary_img,neigh):
    """
    Creates the border for a 3D image
    """
    binary_map = np.asarray(binary_img, dtype=np.uint8)
    neigh = neigh
    west = ndimage.shift(binary_map, [-1, 0,0], order=0)
    east = ndimage.shift(binary_map, [1, 0,0], order=0)
    north = ndimage.shift(binary_map, [0, 1,0], order=0)
    south = ndimage.shift(binary_map, [0, -1,0], order=0)
    top = ndimage.shift(binary_map, [0, 0, 1], order=0)
    bottom = ndimage.shift(binary_map, [0, 0, -1], order=0)
    cumulative = west + east + north + south + top + bottom
    border = ((cumulative < 6) * binary_map) == 1
    return border

def border_distance(ref,seg):
    """
    This functions determines the map of distance from the borders of the
    segmentation and the reference and the border maps themselves
    """
    neigh=8
    border_ref = border_map(ref,neigh)
    border_seg = border_map(seg,neigh)
    oppose_ref = 1 - ref
    oppose_seg = 1 - seg
    # euclidean distance transform
    distance_ref = ndimage.distance_transform_edt(oppose_ref)
    distance_seg = ndimage.distance_transform_edt(oppose_seg)
    distance_border_seg = border_ref * distance_seg
    distance_border_ref = border_seg * distance_ref
    return distance_border_ref, distance_border_seg#, border_ref, border_seg

def Hausdorff_distance(ref,seg):
    """
    This functions calculates the average symmetric distance and the
    hausdorff distance between a segmentation and a reference image
    :return: hausdorff distance and average symmetric distance
    """
    ref_border_dist, seg_border_dist = border_distance(ref,seg)
    hausdorff_distance = np.max(
        [np.max(ref_border_dist), np.max(seg_border_dist)])
    return hausdorff_distance