import pandas as pd
import numpy as np
import sys
from PIL import Image
import os

def SaveImage(imgstr,imgpath):
    img = list(map(np.uint8, imgstr.split()))
    img = np.array(img).reshape([48, 48])
    img = Image.fromarray(img)
    img.save(imgpath)

out_dir = os.path.join("../Data/fer2013_plus")
if os.path.exists(out_dir):
    pass
else:
    os.makedirs(out_dir)

fer2013 = pd.read_csv(r"./fer2013.csv")

ferplus = pd.read_csv(r"./fer2013new.csv")

if len(fer2013) != len(ferplus):
    print("The length of the two datasets is not equal.")

emotions = {
    '0': 'Neutral',
    '1': 'Happy',
    '2': 'Surprise',
    '3': 'Sadness',
    '4': 'Anger',
    '5': 'Disgust',
    '6': 'Fear',
    '7': 'Contempt'
}

os.makedirs(os.path.join(out_dir,'Training'))
os.makedirs(os.path.join(out_dir,'PublicTest'))
os.makedirs(os.path.join(out_dir,'PrivateTest'))

for i in range(8):
    os.makedirs(os.path.join(out_dir,'Training', emotions[str(i)]))
    os.makedirs(os.path.join(out_dir,'PublicTest', emotions[str(i)]))
    os.makedirs(os.path.join(out_dir,'PrivateTest', emotions[str(i)]))

votes = np.zeros([len(ferplus), 10])
print(len(votes))
for i in range(votes.shape[0]):
    for j in range(votes.shape[1]):
        votes[i][j] = ferplus.iloc[i][j + 2]

t = 1 + sys.float_info.epsilon
for i in range(votes.shape[0]):
    for j in range(votes.shape[1]):
        if votes[i][j] < t:
            votes[i][j] = 0

idx = []
lab = []
img_count = np.zeros(8,dtype=int)
for i in range(votes.shape[0]):
    tmp = votes[i]
    maxval = max(tmp)
    ind = np.argmax(tmp)
    if maxval >= 0.5 * tmp.sum() and ind < 8:
        iname = os.path.join(out_dir, fer2013.iloc[i][2], emotions[str(ind)], fer2013.iloc[i][2] + '_' + str(i)+ '_'+ emotions[str(ind)]+'.png')
        img = fer2013.iloc[i][1]
        SaveImage(img,iname)
        img_count[ind] += 1

for i in range(8):
    print(emotions[str(i)], 'images:', img_count[i])
print("Total images:", img_count.sum())
print('Extraction finished!')