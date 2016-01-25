import numpy as np
import os
import sys
import argparse
import h5py
import scipy.spatial
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../Utils/'))
import locker
from FeatStor import load_feat
from PrintUtil import TicTocPrint

tic_toc_print = TicTocPrint()

BASE_CAFFE_PATH = '/home/rgirdhar/Software/vision/caffe_new/'  # w.r.t yoda

parser = argparse.ArgumentParser(description='Extract Features')
parser.add_argument('-i', '--list', type=str, required=True,
    help='File with list of images to run on')
parser.add_argument('-f', '--featdir', type=str, default='',
    help='Features directory')
parser.add_argument('-o', '--outdir', type=str, default='',
    help='Output directory')
parser.add_argument('-s', '--hashes', type=str, required=True,
    help='Path to h5 file with all the hashes')
parser.add_argument('-r', '--resort', type=int, default=100,
    help='Number of matches to resort using the actual features')

args = vars(parser.parse_args())

with open(args['list']) as f:
  imgslist = f.read().splitlines()

with h5py.File(args['hashes'], 'r') as f:
  hashes = f['hashes'].value

nResort = args['resort']
imid = -1
feat_dim = -1
for impath in imgslist:
  imid += 1
  outfpath = os.path.join(args['outdir'], impath + '.h5')
  if not locker.lock(outfpath):
    continue
  tic_toc_print('Working on ' + impath)

  h = hashes[imid]
  D = scipy.spatial.distance.cdist(h[np.newaxis, :], hashes, 'hamming')
  m = np.argsort(D)

  top_matches = m[0, :nResort]
  actFeat = []
  for tm in top_matches.tolist():
    try:
      feat = load_feat(os.path.join(args['featdir'], imgslist[tm] + '.h5'))
      feat_dim = np.shape(feat)[0]
    except:
      sys.stderr.write('Unable to read feature from %s' % imgslist[tm])
      feat = np.zeros((feat_dim, 1))
    actFeat.append(feat)
  actFeat = np.squeeze(np.array(actFeat))
  qFeat = actFeat[0, :]
  D2 = scipy.spatial.distance.cdist(qFeat[np.newaxis, :], actFeat, 'cosine')
  m2 = np.argsort(D2)
  final = top_matches[m2] + 1  # always store as 1-indexed
  with h5py.File(outfpath, 'w') as f:
    f.create_dataset('matches', data=final, compression="gzip", compression_opts=9)

  locker.unlock(outfpath)
