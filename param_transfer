import paddle
import torch
import os
import numpy as np
import pickle


def train():
    model1 = torch.load('../outputs/2021-09-14/15-29-40/best.pth', map_location=torch.device('cpu'))
    model1 = torch.load('../pretrained/GTA5_source.pth', map_location=torch.device('cpu'))
    model1 = model1['state_dict']
    with open('torch.txt', 'w') as f:
        for keys, values in model1.items():
            f.write(keys +'\t'+str(values.shape)+"\n")

    model2 = paddle.load('init_model.pdparams')
    with open('paddle.txt', 'w') as f:
        for keys, values in model2.items():
            f.write(keys +'\t'+str(values.shape)+"\n")
            
    predict = {}
    for key, value in model1.items():
        if ('num_batches_tracked'  not in key): 
            try:
                x = value.numpy()
            except:
                print(key)
            if not isinstance(x, np.ndarray):
                print(type(x))
                raise key
            if 'running_mean' in key:
                key = key.replace('running_mean', '_mean')
            if 'running_var' in key:
                key = key.replace('running_var', '_variance')

            key = 'backbone.' + key
            assert key in model2, print('current key is {}, it is not in the paddle dict'.format(key))

            predict[key] = x
        
          
    params_output = open('torch_transfer_gta5source.pdparams', 'wb')
    pickle.dump(predict, params_output)

if __name__ == '__main__':
    train()
