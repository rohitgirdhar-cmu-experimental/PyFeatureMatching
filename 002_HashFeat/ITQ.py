import numpy as np
from sklearn.decomposition import PCA

def train(feats, nbits, NITER=50):
  pcaComputer = PCA(n_components=nbits)
  print('Computing PCA')
  pcaComputer.fit(feats)
  feats = pcaComputer.transform(feats)  # can also be done by np.dot(feats, pcaComputer.components_)
  pc = pcaComputer.components_.transpose()
  mean = pcaComputer.mean_
  
  rot_matrix = np.random.randn(nbits, nbits)
  u_matrix, _, _ = np.linalg.svd(rot_matrix)
  rot_matrix = u_matrix[:, :nbits]
  for iter in range(NITER):
    print('Running iter %d' % iter)
    Z = np.dot(feats, rot_matrix)
    UX = np.ones(np.shape(Z)) * -1;
    UX[Z >= 0] = 1
    C = np.dot(UX.transpose(), feats);
    ub, _, ua = np.linalg.svd(C)
    rot_matrix_old = rot_matrix.copy()
    rot_matrix = np.array(ua.transpose().dot(ub.transpose()))
    print('Distance of new matrix: %f' % 
        np.linalg.norm(np.matrix(rot_matrix_old) - np.matrix(rot_matrix)))
  return (mean, pc, rot_matrix)

