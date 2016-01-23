import h5py
import numpy as np

def save_feat(feat, outfpath):
  # save using hdf5
  with h5py.File(outfpath, 'w') as f:
      f.create_dataset('feat', data=feat, compression="gzip", compression_opts=9)


def load_feat(fpath):
  with h5py.File(fpath, 'r') as f:
    feat = np.reshape(f['feat'].value, (-1, 1))
  return feat

