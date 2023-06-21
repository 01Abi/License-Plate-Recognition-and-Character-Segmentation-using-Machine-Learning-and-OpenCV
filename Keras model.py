import os
import tensorflow as tf
from tensorflow import keras

(train_images,train_labels),(test_images,test_labels)=tf.keras.datasets.fashion_mnist.load_data()

train_labels=train_labels[:1000]
test_labels=test_labels[:1000]

train_images=train_images[:1000].astype('float32')/255
test_images=test_images[:1000].astype('float32')/255

train_images=train_images.reshape((train_images.shape[0],28,28,1))
test_images=test_images.reshape((test_images.shape[0],28,28,1))

def create_model():
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=2,padding='same',activation='relu',input_shape=(28,28,1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=2,padding='same',activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
model=create_model()
model.summary()
model.fit(train_images,train_labels,batch_size=64,epochs=100,validation_data=(test_images,test_labels))

from keras.models import model_from_json
json_model=model.to_json()
with open('Licenseplatemodel.json','w') as json_file:
    json_file.write(json_model)
model.save_weights('LicensePlate_weights.h5')
loss,acc=model.evaluate(test_images,test_labels,verbose=2)
