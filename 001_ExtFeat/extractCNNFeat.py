import numpy as np
from PIL import Image
import os
import sys
import h5py
import argparse
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../utils/python/'))
import locker


def save_feat(feat, outfpath):
  # save using hdf5
  with h5py.File(outfpath, 'w') as f:
      f.create_dataset('feat', data=feat, compression="gzip", compression_opts=9)


BASE_CAFFE_PATH = '/home/rgirdhar/Software/vision/caffe_new/'  # w.r.t yoda

parser = argparse.ArgumentParser(description='Extract Features')
parser.add_argument('-m', '--mode', type=str, default='cpu',
    help='Run on [gpu/cpu]')
parser.add_argument('-i', '--list', type=str, required=True,
    help='File with list of images to run on')
parser.add_argument('-d', '--dir', type=str, default='',
    help='Images directory')
parser.add_argument('-o', '--outdir', type=str, default='',
    help='Output directory')
parser.add_argument('-n', '--netdesc', type=str, required=True,
    help='Network prototxt')
parser.add_argument('-q', '--netmodel', type=str, required=True,
    help='Network caffemodel')
parser.add_argument('-l', '--layer', type=str, required=True,
    help='Layer to take features from')

args = vars(parser.parse_args())

sys.path.append(os.path.join(BASE_CAFFE_PATH, 'caffe_' + args['mode'], 'python'))
import caffe
# create caffe transformer
im_ht = 227
im_wd = 227
transformer = caffe.io.Transformer({'data': (1,3,im_ht,im_wd)})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
transformer.set_mean('data', np.array([104.00699,116.66877,122.67892]))
net = caffe.Net(args['netdesc'], args['netmodel'], caffe.TEST)

with open(args['list']) as f:
  imgslist = f.read().splitlines()

for impath in imgslist:
  im = plt.imread(os.path.join(args['dir'], impath))
  outfpath = os.path.join(args['outdir'], impath + '.h5')
  if not locker.lock(outfpath):
    continue
  in_ = transformer.preprocess('data', im)

  net.blobs['data'].reshape(1, *in_.shape)
  net.blobs['data'].data[...] = in_
  net.forward()
  out = net.blobs[args['layer']].data[0]

  save_feat(out, outfpath)

  locker.unlock(outfpath)
