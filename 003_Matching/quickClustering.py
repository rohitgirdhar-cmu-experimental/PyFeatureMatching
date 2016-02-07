## Use the matching computed by runMatching.py to quickly compute the de-duplicated images in the dataset

import numpy as np
import os
import sys
import argparse
import h5py
import scipy.spatial
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../Utils/'))
import locker
from FeatStor import load_feat
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../utils/python/'))
from PrintUtil import TicTocPrint

tic_toc_print = TicTocPrint()

parser = argparse.ArgumentParser(description='quickClustering')
parser.add_argument('-i', '--list', type=str, required=True,
    help='File with list of images to run on')
parser.add_argument('-m', '--matchesdir', type=str, default='',
    help='Directory with matches')
parser.add_argument('-r', '--thresh', type=int, default=0.35,
    help='Threshold for considering a match')
parser.add_argument('-t', '--top', type=str, required=True,
    help='Scores fpath')
parser.add_argument('-n', '--numfilter', type=int, default=10000,
    help='Number of images to filter out from')
parser.add_argument('-o', '--outfpath', type=str, required=True,
    help='File to write output to')

args = vars(parser.parse_args())
nfilter = args['numfilter']
thresh = args['thresh']

with open(args['list']) as f:
  imgslist = f.read().splitlines()
selected = np.zeros((len(imgslist), 1))

with open(args['top']) as f:
  scores = [float(el) for el in f.read().splitlines()]
qimgs = np.argsort(-np.array(scores)).tolist()[:nfilter]

res = []
for qid in qimgs:
  impath = imgslist[qid]
  matchpath = os.path.join(args['matchesdir'], impath + '.h5')
  with h5py.File(matchpath, 'r') as f:
    matches = f['matches'].value.tolist()
  midxs = [imgslist.index(m[0]) for m in matches if float(m[1]) < thresh]
  if selected[qid]:
    selected[midxs] = 1
  else:
    res.append(impath)
    selected[qid] = 1
    selected[midxs] = 1

with open(args['outfpath'], 'w') as f:
  f.write('\n'.join(res))

