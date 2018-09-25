from test.ERS_Python import ERSModule
from test import network
import os
import numpy as np
from skimage import (io, segmentation, feature, transform)
import matplotlib.pyplot as plt
import torch

#im_path = './data/input/87015.jpg'

root_path = '/home/laurent.lejeune/medical-labeling/'
im_path = os.path.join(root_path, 'Dataset00/input-frames/frame_0400.png')


max_size = 500 # Otherwise model takes too much memory

cuda = False
gpu_id = 0

#im_path = './data/input/36046.jpg'

model = network.PixelAffinityNet(nr_channel=128, conv1_size=7, use_canny=True)
model.load_state_dict(torch.load('test/bsds500.pkl',
                                 map_location=lambda storage,
                                 loc: storage))
model.eval()
if(cuda):
    model.cuda(gpu_id)

im = io.imread(im_path)

if(max(im.shape[:2]) > max_size):
    im = (transform.rescale(im, max_size/max(im.shape[:2]))*255).astype('uint8')

h = im.shape[0]
w = im.shape[1]
h, w, ch = im.shape

nC = 600
conn8 = 1
lamb = 0.5
sigma = 5.0

input1 = im.transpose((2, 0, 1))
input1 = np.float32(input1) / 255.0
input1 = np.reshape(input1, [1, ch, h, w])
input1 = torch.from_numpy(input1)

# compute Canny edges
edge = feature.canny(np.mean(im, axis=-1), 50, 100)
edge = 1. - np.float32(edge) / 255.
edge = np.reshape(edge, [1, 1, h, w])
input2 = torch.from_numpy(edge)
inputs = torch.cat((input1, input2), 1)
if(cuda):
    inputs = inputs.cuda(gpu_id, non_blocking=True)


# inference
out_x = model(inputs)
inputs_t = torch.transpose(inputs, 2, 3)
out_y_t = model(inputs_t)
out_y = torch.transpose(out_y_t, 2, 3)
outputs = torch.cat((out_x, out_y), 1)

# Compute superpixels
affinity = outputs[0].data.cpu().numpy()
affinity_list = affinity.flatten().tolist()
output = np.zeros_like(im)

label_list = ERSModule.ERSWgtOnly(affinity_list, h, w, nC, conn8, lamb)
label = np.reshape(np.asarray(label_list), (h, w), order='C')

cont_label = segmentation.find_boundaries(
    label, mode='thick')

im_with_conts = im.copy()

im_with_conts[cont_label, :] = (255, 0, 0)

plt.subplot(121); plt.imshow(im);plt.subplot(122);
plt.imshow(im_with_conts);plt.show()

print('Num. of superpixels: {}'.format(np.unique(label).size))
