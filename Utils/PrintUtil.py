import time

class TicTocPrint(object):
  def __init__(self):
    self._last_print = -1
  def __call__(self, msg, n=1):
    if time.time() - self._last_print > n:
      print(msg)
      self._last_print = time.time()
