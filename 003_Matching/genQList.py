import numpy as np

scoresfpath = '/home/rgirdhar/Work/Data/014_TVShows/processed/Scratch/002_FriendsDVD/007_SceneClassScore/002_PlaceNetFT/iter_7K/AllScores.txt'
imgslistfpath = '/home/rgirdhar/Work/Data/014_TVShows/processed/Lists/friends/AllFrames.txt'
outfpath = '/home/rgirdhar/Work/Data/014_TVShows/processed/Scratch/002_FriendsDVD/007_SceneClassScore/002_PlaceNetFT/iter_7K/Queries.txt'
nquery = 20000

with open(imgslistfpath) as f:
  imgslist = f.read().splitlines()
with open(scoresfpath) as f:
  scoreslist = [float(el) for el in f.read().splitlines()]

order = np.argsort(-np.array(scoreslist))  # for 1-index output
imgslist = np.array(imgslist)
np.savetxt(outfpath, imgslist[order[:nquery]], delimiter='\n', fmt='%s')
