"""
CNN to classify healthy vs sick cells
Dataset obtained from ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/cell_images.zip
"""

import numpy as np
#Set the `numpy` pseudo-random generator at a fixed value
#This helps with repeatable results everytime you run the code. 

np.random.seed(1000)

import matplotlib.pyplot as plt
import os
import keras
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential, load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

os.environ['KERAS_BACKEND'] = 'tensorflow' # Added to set the backend as Tensorflow

images_directory = 'C:/Users/Vladis/Desktop/cell_images/'
SIZE = 64
dataset = []
label = []

parasitized_images = os.listdir(images_directory + "Parasitized2")
uninfected_images = os.listdir(images_directory + "Uninfected2")

def create_dataset_and_labels():
    #Iterate through all images in Parasitized folder, resize to 64 x 64
    #Then save as numpy array with name 'dataset'
    #Set the label to this as 0
    
    for i, image_name in tqdm(enumerate(parasitized_images), total=len(parasitized_images)):
        if (image_name.split('.')[1] == "png"):
            image = load_img(images_directory + "Parasitized2/" + image_name, color_mode='rgb', target_size=(SIZE,SIZE))
            dataset.append(np.array(image))
            label.append(0)
            
    #Iterate through all images in Uninfected folder, resize to 64 x 64
    #Then save as numpy array with name 'dataset'
    #Set the label to this as 1
    
    for i, image_name in tqdm(enumerate(uninfected_images), total=len(uninfected_images)):
        if (image_name.split('.')[1] == "png"):
            image = load_img(images_directory + "Uninfected2/" + image_name, color_mode='rgb', target_size=(SIZE,SIZE))
            dataset.append(np.array(image))
            label.append(1)
        
        
def build_model():
    INPUT_SHAPE = (SIZE, SIZE, 3)
    inp = keras.layers.Input(shape=INPUT_SHAPE)
    conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), 
                                   activation='relu', padding='same')(inp)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    norm1 = keras.layers.BatchNormalization(axis = -1)(pool1)
    drop1 = keras.layers.Dropout(rate=0.2)(norm1)
    conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3), 
                                   activation='relu', padding='same')(drop1)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    norm2 = keras.layers.BatchNormalization(axis = -1)(pool2)
    drop2 = keras.layers.Dropout(rate=0.2)(norm2)
    
    flat = keras.layers.Flatten()(drop2)    #Flatten the matrix to get it ready for dense.
    
    hidden1 = keras.layers.Dense(512, activation='relu')(flat)
    norm3 = keras.layers.BatchNormalization(axis = -1)(hidden1)
    drop3 = keras.layers.Dropout(rate=0.2)(norm3)
    hidden2 = keras.layers.Dense(256, activation='relu')(drop3)
    norm4 = keras.layers.BatchNormalization(axis = -1)(hidden2)
    drop4 = keras.layers.Dropout(rate=0.2)(norm4)
    
    out = keras.layers.Dense(2, activation='sigmoid')(drop4)   #units=1 gives error
    
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_sequential_model():    
    model = None
    model = Sequential([
        Convolution2D(
            filters= 32, 
            kernel_size=(3, 3), 
            input_shape = (SIZE, SIZE, 3), 
            activation = 'relu', 
            padding='same',
            data_format='channels_last',
        ),
        MaxPooling2D(pool_size = (2, 2), data_format="channels_last"),
        BatchNormalization(axis = -1),
        Dropout(rate=0.2),
        Convolution2D(filters= 32, kernel_size=(3, 3), activation = 'relu',padding='same'),
        MaxPooling2D(pool_size = (2, 2), data_format="channels_last"),
        BatchNormalization(axis = -1),
        Dropout(rate=0.2),
        Flatten(),
        Dense(activation = 'relu', units=512),
        BatchNormalization(axis = -1),
        Dropout(rate=0.2),
        Dense(activation = 'relu', units=256),
        BatchNormalization(axis = -1),
        Dropout(rate=0.2),
        Dense(activation = 'sigmoid', units=2),
    ])
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t = f.suptitle('CNN Performance', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)
    
    max_epoch = len(history.history['accuracy'])+1
    epoch_list = list(range(1,max_epoch))
    ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
    ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(1, max_epoch, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")
    
    ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
    ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(1, max_epoch, 5))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")

def run_test():
    create_dataset_and_labels()
    X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size = 0.20, random_state = 0)
    model = build_sequential_model()
    print(model.summary())
    history = model.fit(
            x=np.array(X_train),
            y=y_train ,
            batch_size=64,
            verbose=1,
            epochs=25,
            validation_split=0.1,
            shuffle=False
            )
    print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(y_test))[1]*100))
    summarize_diagnostics(history)
    model.save('malaria_cnn.h5')
    
def load_image(filename):
	# load the image
	img = load_img(filename, color_mode='rgb', target_size=(SIZE,SIZE))
	# convert to np.array
	img = np.array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, img.shape[0],img.shape[1],img.shape[2])
	return img
 
# load an image and predict the class
def run_example():
	# load the image
	img = load_image('C:/Users/Vladis/Desktop/cell_images/Uninfected/C241NThinF_IMG_20151207_124643_cell_125.png')
	# load model
	model = load_model('malaria_cnn.h5')
	# predict the class
	result = model.predict(img)
	print('Prediction:', result[0])
run_example()
