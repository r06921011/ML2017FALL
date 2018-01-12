import sys
import numpy as np
import pandas as pd
from skimage import io
from skimage import transform

def read_data(fileName):
    data = io.imread(fileName)
    image = np.asarray(data)
    image = image.flatten()
    return image

im = []
path = sys.argv[1]
data_num = 415
for idx in range(data_num):

	data = read_data(path + '/' + str(idx) + '.jpg')
	im.append(data)
	print('append image',idx)

ima = np.asarray(im)
ima_mean = np.mean(ima, axis=0)
imag = ima - ima_mean
print('X=',imag.shape)

U, s, V = np.linalg.svd(imag.T, full_matrices=False)
print('U.shape',U.shape)

w = np.dot(imag, U)

Name = sys.argv[2]
re_idx = int(Name[:Name.find('.')])

recon_num = re_idx
recon = ima_mean + np.dot(w[recon_num, 0:4], U[:, 0:4].T)

recon -= np.min(recon)
recon /= np.max(recon)
recon = (recon*255).astype(np.uint8)
print('save reconstruction image',recon_num)
io.imsave('reconstruction.jpg', recon.reshape((600, 600, 3)))
