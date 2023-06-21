#Import Necessary Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
import glob
from PIL import Image

#Loading of the model

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)
wpod_net_path = "Licenseplatemodel.json"
wpod_net = load_model(wpod_net_path)

"""def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img"""

#Preprocessing of an image
#Convert the image to grayscale
#img=Image.open(r"C:\Users\User\Desktop\SEMESTER8\Image Processing Laboratory\License Plate Detection\License Plate Dataset\china_car_plate.jpg")
#gray_image=img.convert('L')
#gray_image.save('china_car_plate_gray.jpg')

#Increase Brightness and Contrast of an image

#image_path=Image.open(r"C:\Users\User\Desktop\SEMESTER8\Image Processing Laboratory\License Plate Detection\License Plate Dataset\china_car_plate.jpg")
#img=cv2.imread(image_path)

#alpha=1.5 #Contrast Control
#beta=10 #Brightness Control

#calling of convertScaleAbsFunction
#img = cv2.resize(img, (224,224))

#adjusted=cv2.convertScaleAbs(img,alpha=alpha,beta=beta)
#adjusted.save('adjusted_image.jpg')



