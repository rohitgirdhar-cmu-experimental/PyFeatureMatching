import numpy as np
import os
import sys
import argparse
import numpy as np
import random
random.seed(1)
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../utils/python/'))
import locker
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../Utils/'))
from FeatStor import load_feat
from PrintUtil import TicTocPrint
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import ITQ

tic_toc_print = TicTocPrint()

parser = argparse.ArgumentParser(description='Extract Features')
parser.add_argument('-i', '--list', type=str, required=True,
    help='File with list of images')
parser.add_argument('-d', '--dir', type=str, default='',
    help='Features directory')
parser.add_argument('-o', '--outpath', type=str, required=True,
    help='Output hdf5 file path to store the learnt parameters')
parser.add_argument('-n', '--numfeat', type=int, default=100000,
    help='Number of features to use for learning ITQ')

args = vars(parser.parse_args())

outfpath = args['outpath']
if not outfpath.endswith('.h5'):
  outfpath += '.h5'

with open(args['list']) as f:
  imgslist = f.read().splitlines()

random.shuffle(imgslist)

nFeat = args['numfeat']
allfeats = []
for impath in imgslist:
  tic_toc_print('Read %d features' % len(allfeats))
  try:
    featpath = os.path.join(args['dir'], impath + '.h5')
    feat = load_feat(featpath).transpose()
    allfeats.append(feat)
  except:
    continue
  if len(allfeats) >= nFeat:
    break

allfeats = np.squeeze(np.array(allfeats))
allfeats[np.isnan(allfeats)] = 0
allfeats[np.isinf(allfeats)] = 0
mean, pc, R = ITQ.train(allfeats, 12)
with h5py.File(outfpath) as f:
  f.create_dataset('R', data=R, compression="gzip", compression_opts=9)
  f.create_dataset('pc', data=pc, compression="gzip", compression_opts=9)
  f.create_dataset('mean', data=mean, compression="gzip", compression_opts=9)
