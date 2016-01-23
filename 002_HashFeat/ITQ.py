import numpy as np
from scipy import linalg as LA
import h5py
import os

PCA_CACHE_FPATH = 'pca.cache.h5'
def train(feats, nbits, NITER=50):
  print('Computing PCA')
  if os.path.exists(PCA_CACHE_FPATH):
    with h5py.File(PCA_CACHE_FPATH, 'r') as f:
      pc = f['pc'].value
      mean = f['mean'].value
    feats = transformPCA(feats, pc, mean)
  else:
    feats, pc, mean = PCA(feats, nbits)
    with h5py.File(PCA_CACHE_FPATH, 'w') as f:
      f.create_dataset('pc', data=pc)
      f.create_dataset('mean', data=mean)
  
  rot_matrix = np.random.randn(nbits, nbits)
  u_matrix, _, _ = np.linalg.svd(rot_matrix)
  rot_matrix = u_matrix[:, :nbits]
  for iter in range(NITER):
    print('Running iter %d' % iter)
    Z = np.dot(feats, rot_matrix)
    UX = np.ones(np.shape(Z)) * -1
    UX[Z >= 0] = 1
    C = np.dot(UX.transpose(), feats)
    ub, _, ua = np.linalg.svd(C)
    rot_matrix_old = rot_matrix.copy()
    rot_matrix = np.array(ua.transpose().dot(ub.transpose()))
    print('Distance of new matrix: %f' % 
        np.linalg.norm(np.matrix(rot_matrix_old) - np.matrix(rot_matrix)))
  return (mean, pc, rot_matrix)

def PCA(data, dims_rescaled_data=2):
  # from http://stackoverflow.com/a/13224592/1492614
  """
  returns: data transformed in 2 dims/columns + regenerated original data
  pass in: data as 2D NumPy array
  """
  m, n = data.shape
  # mean center the data
  mean_ = np.mean(data, axis=0)
  data -= mean_
  # calculate the covariance matrix
  R = np.cov(data, rowvar=False)
  # calculate eigenvectors & eigenvalues of the covariance matrix
  # use 'eigh' rather than 'eig' since R is symmetric, 
  # the performance gain is substantial
  R[np.isnan(R)] = 0
  R[np.isinf(R)] = 0
  evals, evecs = LA.eigh(R)
  # sort eigenvalue in decreasing order
  idx = np.argsort(evals)[::-1]
  evecs = evecs[:,idx]
  # sort eigenvectors according to same index
  evals = evals[idx]
  # select the first n eigenvectors (n is desired dimension
  # of rescaled data array, or dims_rescaled_data)
  evecs = evecs[:, :dims_rescaled_data]
  # carry out the transformation on the data using eigenvectors
  # and return the re-scaled data, eigenvalues, and eigenvectors
  #return np.dot(evecs.T, data.T).T, eigenvalues, eigenvectors
  return np.dot(data, evecs), evecs, mean_

def transformPCA(feats, pc, mean):
  feats -= mean
  feats = np.dot(feats, pc)
  return feats

