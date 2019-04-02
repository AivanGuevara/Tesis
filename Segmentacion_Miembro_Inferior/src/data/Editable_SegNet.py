import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


import pandas as pd

import sys

from skimage.io import imread
from matplotlib import pyplot as plt

import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=gpu0, floatX=float32, optimizer=fast_compile'

#from keras.models import Sequential, Model
from keras import models
from keras.optimizers import SGD
#from keras.optimizers import RMSprop, SGD


#%% 

path = 'Data2/'
img_w = 512
img_h = 512
n_labels = 4

n_train = 400
n_test = 40
#%%







def label_map(labels):
    label_map = np.zeros([img_h, img_w, n_labels])    
    for r in range(img_h):
        for c in range(img_w):
            label_map[r, c, labels[r][c]] = 1
    return label_map

def prep_data(mode):
    assert mode in {'test', 'train'}, \
        'mode should be either \'test\' or \'train\''
    data = []
    label = []
    df = pd.read_csv(path + mode + '.csv')
    n = n_train if mode == 'train' else n_test
    for i, item in df.iterrows():
        if i >= n:
            break
        img, gt = [imread(path + item[0])], np.clip(imread(path + item[1]), 0, 1)
        data.append(img)
        label.append(label_map(gt))
        sys.stdout.write('\r')
        sys.stdout.write(mode + ": [%-20s] %d%%" % ('=' * int(20. * (i + 1) / n - 1) + '>',
                                                    int(100. * (i + 1) / n)))
        sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    data, label = np.array(data).reshape((n, img_h,img_w, 1)), np.array(label).reshape((n, img_h * img_w, n_labels))

    print (mode + ': OK')
    print ('\tshapes: {}, {}'.format(data.shape, label.shape))
    print ('\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    print ('\tmemory: {}, {} MB'.format(data.nbytes / 1048576, label.nbytes / 1048576))

    return data, label

def plot_results(output):
    gt = []
    df = pd.read_csv(path + 'test.csv')
    for i, item in df.iterrows():
        gt.append(np.clip(imread(path + item[1]), 0, 1))

    plt.figure(figsize=(15, 2 * n_test))
    for i, item in df.iterrows():
        plt.subplot(n_test, 4, 4 * i + 1)
        plt.title('Ground Truth')
        plt.axis('off')
        gt = imread(path + item[1])
        plt.imshow(np.clip(gt, 0, 1))

        plt.subplot(n_test, 4, 4 * i + 2)
        plt.title('Prediction')
        plt.axis('off')
        labeled = np.argmax(output[i], axis=-1)
        plt.imshow(labeled)

        plt.subplot(n_test, 4, 4 * i + 3)
        plt.title('Heat map')
        plt.axis('off')
        plt.imshow(output[i][:, :, 1])

        plt.subplot(n_test, 4, 4 * i + 4)
        plt.title('Comparison')
        plt.axis('off')
        rgb = np.empty((img_h, img_w, 3))
        rgb[:, :, 0] = labeled
        rgb[:, :, 1] = imread(path + item[0])
        rgb[:, :, 2] = gt
        plt.imshow(rgb)

    plt.savefig('result.png')
    plt.show()
#%%


#%%
#########################################################################################################

#with open('/home/duilio/Documentos/Tesis/Repositorios_GitHub/Segnet/keras-segnet/editable_model_5l.json') as model_file:
with open('editable_model_4l.json') as model_file:
    autoencoder = models.model_from_json(model_file.read())

optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)

#######hiperparametros del Curso de Juan-------------------------------------------------------------------------------------
########learning_rate = 0.005
########batch_size = 25
########num_epochs = 25
########steps_per_epoch = 500
########validation_steps = 50
########workers = 2
########model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')
#---------------------------------------------------------------------------------------------------------------------------
autoencoder.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
print ('Compiled: OK')
#----------------------------------------------------------------------------------------------------------------
#######model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

###model.summary()
#--------------------------------------------------------------------------------------------------------------------

#####model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])

#####cnn_model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=["accuracy"])



#--------------------------------------------------------------------------------

#####Shuffle and Split the dataset

#####x,y = shuffle(img_data,Y, random_state=2)
#####X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#####print("X_train shape = {}".format(X_train.shape))
#####print("X_test shape = {}".format(X_test.shape))


#--------------------------------------------------------------------------------
#%%
train_data, train_label = prep_data('train')

#######Data augmentation
#######x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
#######y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)



nb_epoch = 20
batch_size = 5
history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
#--------------------------------------------------------------------------------------------------------------------------
###### Data iterators for loading the training and validation data
########train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
########                                               data_folder=os.path.join('..', 'data', 'train'),
########                                               image_shape=image_shape,
########                                               shift_aug=True)
########val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
########                                             data_folder=os.path.join('..', 'data', 'validation'),
########                                             image_shape=image_shape)
#########logger_cb = plotting_tools.LoggerPlotter()
#########callbacks = [logger_cb]
########model.fit_generator(train_iter,
########                    steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
########                    epochs = num_epochs, # the number of epochs to train for,
########                    validation_data = val_iter, # validation iterator
########                    validation_steps = validation_steps, # the number of batches to validate on
########                    callbacks=callbacks,
########                    workers = workers)

#-----------------------------------------------------------------------------------------------------------------------------

###early_stopping = EarlyStopping(patience=10, verbose=1)
###model_checkpoint = ModelCheckpoint("./keras.model", save_best_only=True, verbose=1)
###reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)

###epochs = 200
###batch_size = 32

###history = model.fit(x_train, y_train,
###                    validation_data=[x_valid, y_valid], 
###                    epochs=epochs,
###                    batch_size=batch_size,
###                    callbacks=[early_stopping, model_checkpoint, reduce_lr])
#---------------------------------------------------------------------------------------------------------------------------
# number of samples used for determining the samples_per_epoch
###nb_train_samples = 65
###nb_validation_samples = 10
###epochs = 20
###batch_size = 5  

###train_datagen = ImageDataGenerator(
###        rescale=1./255,            # normalize pixel values to [0,1]
###        shear_range=0.2,      
###        zoom_range=0.2,    
###        rotation_range=20,
###        width_shift_range=0.2,
###        height_shift_range=0.2,
###        horizontal_flip=True)  


###val_datagen = ImageDataGenerator(
###         rescale=1./255)       # normalize pixel values to [0,1]

###train_generator = train_datagen.flow_from_directory(
###    train_data_dir,
###    target_size=(img_height, img_width),
###    batch_size=batch_size,
###    class_mode='binary')

###validation_generator = train_datagen.flow_from_directory(
###    validation_data_dir,
###    target_size=(img_height, img_width),
###    batch_size=batch_size,
###    class_mode='binary')
#---------------------------------------------------------------------------------------------
####from keras.preprocessing.image import ImageDataGenerator
####dg_args = dict(featurewise_center = False, 
####                  samplewise_center = False,
####                  rotation_range = 5, 
###                  width_shift_range = 0.05, 
###                  height_shift_range = 0.05, 
###                  shear_range = 0.01,
###                  zoom_range = [0.8, 1.2],  
###               # anatomically it doesnt make sense, but many images are flipped
###                  horizontal_flip = True,  
###                  vertical_flip = False,
###                  fill_mode = 'nearest',
###               data_format = 'channels_last')

###image_gen = ImageDataGenerator(**dg_args)
###-----------------------------------------------------------------------








autoencoder.save_weights('model_5l_weight_ep50.hdf5')


#autoencoder.load_weights('model_5l_weight_ep50.hdf5')

#--------------Dibujar las curvas de Perdida y prediccion------------------------------------

######fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
######ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
######ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
######ax_acc.plot(history.epoch, history.history["acc"], label="Train accuracy")
######ax_acc.plot(history.epoch, history.history["val_acc"], label="Validation accuracy")










test_data, test_label = prep_data('test')
score = autoencoder.evaluate(test_data, test_label, verbose=0)
print ('Test score:', score[0])
print ('Test accuracy:', score[1])

output = autoencoder.predict_proba(test_data, verbose=0)
output = output.reshape((output.shape[0], img_h, img_w, n_labels))

plot_results(output)

