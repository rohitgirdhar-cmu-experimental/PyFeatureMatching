import h5py

def save_feat(feat, outfpath):
  # save using hdf5
  with h5py.File(outfpath, 'w') as f:
      f.create_dataset('feat', data=feat, compression="gzip", compression_opts=9)

