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

# NOTE: It stores the (id,score) with id 0-indexed. Fix it in future!

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
parser.add_argument('-t', '--top', type=str, default='',
    help='[optional] path to scores for queries, so select only those with high scores')
parser.add_argument('-q', '--qlist', type=str, default='',
    help='[optional] List of query images. Takes precedence over --top')
parser.add_argument('-n', '--nqueries', type=int, default=-1,
    help='[optional] Number of images to run as queries. By default (-1) => all')

args = vars(parser.parse_args())

with open(args['list']) as f:
  imgslist = f.read().splitlines()

qimgs = range(len(imgslist))
if len(args['qlist']) > 0:
  qimgs = []
  with open(args['qlist']) as f:
    for line in f:
      try:
        pos = imgslist.index(line)
        qimgs.append(pos)
      except:
        print 'Skipping. Cant find query item:', line
elif len(args['top']) > 0:
  with open(args['top']) as f:
    scores = [float(el) for el in f.read().splitlines()]
  qimgs = np.argsort(-np.array(scores)).tolist()

if args['nqueries'] >= 0:
  qimgs = qimgs[:args['nqueries']]

with h5py.File(args['hashes'], 'r') as f:
  hashes = f['hashes'].value

nResort = args['resort']
feat_dim = -1
for qid in qimgs:
  impath = imgslist[qid]
  outfpath = os.path.join(args['outdir'], impath + '.h5')
  if not locker.lock(outfpath):
    continue
  tic_toc_print('Working on ' + impath)

  h = hashes[qid]
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
  final = zip(top_matches[m2].tolist()[0], D2[0,m2].astype('float').tolist()[0])  # always store as 1-indexed
  with h5py.File(outfpath, 'w') as f:
    f.create_dataset('matches', data=final, compression="gzip", compression_opts=9)

  locker.unlock(outfpath)
