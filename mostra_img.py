import cv2
import pandas as pd

#unet-train1.py
#Treina rede unet para segmentacao semantica de eliret
from PIL import Image
import pandas as pd
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import cv2; import numpy as np; np.random.seed(7); import sys
import tensorflow.keras as keras; from tensorflow.keras.models import *
from tensorflow.keras.layers import *; from tensorflow.keras.optimizers import *
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean

tamanho_img=512

def leCsv(nomeDir,nomeArq):
  print("Lendo: ",nomeArq); arq=open(os.path.join(nomeDir,nomeArq),"r")
  lines=arq.readlines(); arq.close(); n=len(lines)
  nl,nc = 286,384
  AX=np.empty((n,nl,nc),dtype='uint8'); AY=np.empty((n,1,4),dtype='uint8')
  i=0
  for linha in lines:
    linha=linha.strip('\n'); linha=linha.split(';')
    nomeDir=''
    #print(os.path.join(nomeDir,linha[0]))
    AX[i]=Image.open(os.path.join(nomeDir,linha[0]))
    AX[i]= np.array(AX[i])
  #  print(AX[i])
    print(linha)
  #  AX[i]=AX[i]*255/AX[i].max()
    AY_aux=str(pd.read_csv(os.path.join(nomeDir,linha[1])))

    AY_aux= AY_aux.split('\n')[1]
    AY_aux=AY_aux.split('\\t')
    AY_aux[0]=AY_aux[0].split(' ')[2]
    print(AY_aux)
    AY[i]= np.array(AY_aux)
 #   AY[ AY>=125 ] = 255
 #   AY[ AY<125] = 0
 #   AY[i]=AY[i]*255/AY[i].max()
 #   f = plt.figure()
 #   f.add_subplot(1,4,1); plt.imshow(AX[i],cmap="gray"); plt.axis('off')
 #   f.add_subplot(1,4,2); plt.imshow(AY[i],cmap="gray"); plt.axis('off')
 #   plt.show(block=True)
 #   ax1=np.float32(AX[i].reshape(nl,nc))/255.0
 #   ax2=np.float32(AY[i].reshape(nl,nc))/255.0
 #   f = plt.figure()
 #   f.add_subplot(1,4,1); plt.imshow(ax1,cmap="gray"); plt.axis('off')
 #   f.add_subplot(1,4,2); plt.imshow(ax2,cmap="gray"); plt.axis('off')
 #   plt.show(block=True)
    i=i+1

  ax= np.float32(AX)/255.0
  ay= np.float32(AY)/255.0 #Entre 0 e +1
 #   AY[i]=AY[i]*255/AY[i].max()

  return ax,ay

#<<<<<<<<<<<<<<<<<<<< main <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
bdDir = ""
ax, ay = leCsv(bdDir,"train.csv")
vx, vy = leCsv(bdDir,"test.csv")
#qx, qy = leCsv(bdDir,"teste.csv")
outDir = "."; os.chdir(outDir)


img=cv2.imread('BioID-FaceDatabase-V1/BioID_0000.pgm')
square=str(pd.read_csv('BioID-FaceDatabase-V1/BioID_0000.eye'))
a= square.split('\n')[1]
b=a.split('\\t')
print(b)
b[0]=b[0].split(' ')[2]
print(b)
cv2.rectangle(img, (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), (255,255,255) ,2)
cv2.imshow('image',img)

cv2.waitKey()




