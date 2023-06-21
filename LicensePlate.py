import cv2
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json
import glob
from PIL import Image
import PIL
from numpy.linalg import norm

#save_filename='output_img.jpg'
img=np.array(Image.open(r"C:\Users\User\Desktop\SEMESTER8\Image Processing Laboratory\License Plate Detection\License Plate Dataset\china_car_plate.jpg"))

#alpha=1.5 #Contrast Control
#beta=10 #Brightness Control

#adjusted=cv2.convertScaleAbs(img,alpha=alpha,beta=beta)
#cv2.imwrite(r"C:\Users\User\Desktop\SEMESTER8\Image Processing Laboratory\License Plate Detection\License Plate Dataset\adjusted.jpg",adjusted)

#contrast=5
#brightness=2
#output=cv2.addWeighted(img,contrast,img,0,brightness)
#cv2.imwrite(r"C:\Users\User\Desktop\SEMESTER8\Image Processing Laboratory\License Plate Detection\License Plate Dataset\adjusted1.jpg",output)

#if len(adjusted.shape)==3:
    #print(np.average(norm(adjusted,axis=2)))
#else:
    #print(np.average(adjusted))
#L,A,B=cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2LAB))
#L=L/np.max(L)
#print(np.mean(L))
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
colors=("red","green","blue")
plt.figure()
plt.xlim([0,256])
for channel_id,color in enumerate(colors):
    histogram,bin_edges=np.histogram(img1[:,:,channel_id],bins=256,range=(0,256))
    plt.plot(bin_edges[0:-1],histogram,color=color)

plt.title("Color Histogram")
plt.xlabel=("Color Value")
plt.ylabel("Pixel Count")
plt.show()



                                                         