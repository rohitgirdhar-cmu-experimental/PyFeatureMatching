import h5py
import numpy as np
import scipy.ndimage

def save_feat(feat, outfpath):
  # save using hdf5
  with h5py.File(outfpath, 'w') as f:
      f.create_dataset('feat', data=feat, compression="gzip", compression_opts=9)


def load_feat(fpath, frac=1):
  with h5py.File(fpath, 'r') as f:
    feat = f['feat'].value
  feat = scipy.ndimage.zoom(feat, frac)
  feat = np.reshape(feat, (-1,1))
  return feat

