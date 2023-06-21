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
wpod_net_path ="Licenseplatemodel.json"
wpod_net=load_model(r"C:\Users\User\Desktop\SEMESTER8\Image Processing Laboratory\License Plate Detection\Licenseplatemodel.json")

def preprocess_image(image_path,resize=False):
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
image_paths = glob.glob("License_Plate_Dataset/*.jpg")
print("Found %i images..."%(len(image_paths)))





