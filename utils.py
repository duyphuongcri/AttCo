import numpy as np
import torch

def post_processing(model, input, stride=(27, 112, 112), target_size=(128,128,128)):
    count_arr = torch.zeros_like(input)
    prob_arr = torch.zeros_like(input)
    for z in range(int(np.ceil(count_arr.shape[0]/target_size[0]))):
        for x in range(int(np.ceil(count_arr.shape[1]/target_size[1]))):
            for y in range(int(np.ceil(count_arr.shape[2]/target_size[2]))):
                patch = input[:, :, z*stride[0]:z*stride[0] + target_size[0], x*stride[1]:x*stride[1]+target_size[1], y*stride[2]:y*stride[2]+target_size[2]]
                output = model(patch)
                output = torch.softmax(output, dim=1)
                prob_arr[:, :, z*stride[0]:z*stride[0] + target_size[0], x*stride[1]:x*stride[1]+target_size[1], y*stride[2]:y*stride[2]+target_size[2]] += output
                count_arr[:, :, z*stride[0]:z*stride[0] + target_size[0], x*stride[1]:x*stride[1]+target_size[1], y*stride[2]:y*stride[2]+target_size[2]] += 1
    prob_arr = prob_arr / count_arr
    return prob_arr