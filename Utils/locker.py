import os

def lock(fpath):
  lock_fpath = fpath + '.lock'
  if os.path.exists(fpath) or os.path.exists(lock_fpath):
    return False
  try:
    os.makedirs(lock_fpath)
    return True
  except:
    return False

def unlock(fpath):
  lock_fpath = fpath + '.lock'
  try:
    os.rmdir(lock_fpath)
  except:
    pass
