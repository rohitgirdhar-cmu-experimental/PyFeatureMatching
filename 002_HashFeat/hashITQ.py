import numpy as np
import os
import sys
import argparse
import numpy as np
import h5py
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
parser.add_argument('-p', '--paramfile', type=str, required=True,
    help='Hdf5 file path to store the learnt parameters')
parser.add_argument('-o', '--outpath', type=str, required=True,
    help='Hdf5 file to store output hashes')

args = vars(parser.parse_args())

outfpath = args['outpath']
if not outfpath.endswith('.h5'):
  outfpath += '.h5'

with open(args['list']) as f:
  imgslist = f.read().splitlines()

with h5py.File(args['paramfile'], 'r') as f:
  R = f['R'].value
  pc = f['pc'].value
  mean = f['mean'].value

nDone = 0
allhashes = []
for impath in imgslist:
  tic_toc_print('Done %d features' % nDone)
  try:
    featpath = os.path.join(args['dir'], impath + '.h5')
    feat = load_feat(featpath).transpose()
    hash = ITQ.hash(feat, pc, mean, R)
    allhashes.append(hash)
    nDone += 1
  except:
    continue

allhashes = np.squeeze(np.array(allhashes)).astype('bool')
with h5py.File(args['outpath'], 'w') as f:
  f.create_dataset('hashes', data=allhashes, compression="gzip", compression_opts=9)

