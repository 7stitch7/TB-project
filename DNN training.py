#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import scipy
import skimage
from skimage.transform import resize

# In[2]:


from numpy.random import seed

seed(101)
import tensorflow

tensorflow.random.set_seed(101)

import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import binary_accuracy

import os
import cv2

import imageio
import skimage
import skimage.io
import skimage.transform

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')



# Total number of images we want to have in each class
NUM_AUG_IMAGES_WANTED = 1000

# We will resize the images
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150

# In[4]:


os.listdir('./TB data open source')

# In[5]:


print(len(os.listdir('./TB data open source/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png')))
print(len(os.listdir('./TB data open source/Montgomery/MontgomerySet/CXR_png')))

# In[4]:


shen_image_list = os.listdir('./TB data open source/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png')
mont_image_list = os.listdir('./TB data open source/Montgomery/MontgomerySet/CXR_png')




# put the images into dataframes
df_shen = pd.DataFrame(shen_image_list, columns=['image_id'])
df_mont = pd.DataFrame(mont_image_list, columns=['image_id'])

# remove the 'Thunbs.db' line
df_shen = df_shen[df_shen['image_id'] != 'Thumbs.db']
df_mont = df_mont[df_mont['image_id'] != 'Thumbs.db']

# Reset the index or this will cause an error later
df_shen.reset_index(inplace=True, drop=True)
df_mont.reset_index(inplace=True, drop=True)

print(df_shen.shape)
print(df_mont.shape)

# In[13]:


df_shen.head()

# In[14]:


df_mont.head()


# In[6]:


# Function to select the 4th index from the end of the string (file name)
# example: CHNCXR_0470_1.png --> 1 is the label, meaning TB is present.

def extract_target(x):
    target = int(x[-5])
    if target == 0:
        return 'Normal'
    if target == 1:
        return 'Tuberculosis'


# In[7]:


# Assign the target labels

df_shen['target'] = df_shen['image_id'].apply(extract_target)

df_mont['target'] = df_mont['image_id'].apply(extract_target)

# In[8]:


# Shenzen Dataset

df_shen['target'].value_counts()

# In[9]:


# Montgomery Dataset

df_mont['target'].value_counts()


# In[10]:


def draw_category_images(col_name, figure_cols, df, IMAGE_PATH):
    """
    Give a column in a dataframe,
    this function takes a sample of each class and displays that
    sample on one row. The sample size is the same as figure_cols which
    is the number of columns in the figure.
    Because this function takes a random sample, each time the function is run it
    displays different images.
    """

    categories = (df.groupby([col_name])[col_name].nunique()).index
    f, ax = plt.subplots(nrows=len(categories), ncols=figure_cols,
                         figsize=(4 * figure_cols, 4 * len(categories)))  # adjust size here
    # draw a number of images for each location
    for i, cat in enumerate(categories):
        sample = df[df[col_name] == cat].sample(figure_cols)  # figure_cols is also the sample size
        for j in range(0, figure_cols):
            file = IMAGE_PATH + sample.iloc[j]['image_id']
            im = imageio.imread(file)
            ax[i, j].imshow(im, resample=True, cmap='gray')
            ax[i, j].set_title(cat, fontsize=14)
    plt.tight_layout()
    plt.show()




def read_image_sizes(file_name):
    """
    1. Get the shape of the image
    2. Get the min and max pixel values in the image.
    Getting pixel values will tell if any pre-processing has been done.
    3. This info will be added to the original dataframe.
    """
    image = cv2.imread(IMAGE_PATH + file_name)
    max_pixel_val = image.max()
    min_pixel_val = image.min()

    # image.shape[2] represents the number of channels: (height, width, num_channels).
    # Here we are saying: If the shape does not have a value for num_channels (height, width)
    # then assign 1 to the number of channels.
    if len(image.shape) > 2:  # i.e. more than two numbers in the tuple
        output = [image.shape[0], image.shape[1], image.shape[2], max_pixel_val, min_pixel_val]
    else:
        output = [image.shape[0], image.shape[1], 1, max_pixel_val, min_pixel_val]
    return output





IMAGE_PATH = './TB data open source/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png/'
#
# m = np.stack(df_shen['image_id'].apply(read_image_sizes))
# df = pd.DataFrame(m, columns=['w', 'h', 'c', 'max_pixel_val', 'min_pixel_val'])
# df_shen = pd.concat([df_shen, df], axis=1, sort=False)
#
#
#
# # In[14]:
#
#
# IMAGE_PATH = './TB data open source/Montgomery/MontgomerySet/CXR_png/'
#
# m = np.stack(df_mont['image_id'].apply(read_image_sizes))
# df = pd.DataFrame(m, columns=['w', 'h', 'c', 'max_pixel_val', 'min_pixel_val'])
# df_mont = pd.concat([df_mont, df], axis=1, sort=False)




### Combine the two dataframes and shuffle

df_data = pd.concat([df_shen, df_mont], axis=0).reset_index(drop=True)

df_data = shuffle(df_data)



# In[16]:


# Create a new column called 'labels' that maps the classes to binary values.
df_data['labels'] = df_data['target'].map({'Normal': 0, 'Tuberculosis': 1})



# train_test_split

y = df_data['labels']

df_train, df_val = train_test_split(df_data, test_size=0.15, random_state=101, stratify=y)

print(df_train.shape)
print(df_val.shape)

# In[18]:


# Create a new directory
base_dir = 'base_dir'
os.mkdir(base_dir)


# [CREATE FOLDERS INSIDE THE BASE DIRECTORY]

# now we create 2 folders inside 'base_dir':

# train
# Normal
# Tuberculosis

# val
# Normal
# Tuberculosis


# create a path to 'base_dir' to which we will join the names of the new folders
# train_dir
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

# val_dir
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)


# [CREATE FOLDERS INSIDE THE TRAIN AND VALIDATION FOLDERS]
# Inside each folder we create seperate folders for each class

# create new folders inside train_dir
Normal = os.path.join(train_dir, 'Normal')
os.mkdir(Normal)
Tuberculosis = os.path.join(train_dir, 'Tuberculosis')
os.mkdir(Tuberculosis)


# create new folders inside val_dir
Normal = os.path.join(val_dir, 'Normal')
os.mkdir(Normal)
Tuberculosis = os.path.join(val_dir, 'Tuberculosis')
os.mkdir(Tuberculosis)


# In[19]:


# Set the image_id as the index in df_data
df_data.set_index('image_id', inplace=True)

# In[20]:


# Get a list of images in each of the two folders
folder_1 = os.listdir('./TB data open source/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png')
folder_2 = os.listdir('./TB data open source/Montgomery/MontgomerySet/CXR_png')

# Get a list of train and val images
train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])

# In[21]:


# Transfer the train images

for image in train_list:

    fname = image
    label = df_data.loc[image, 'target']

    if fname in folder_1:
        # source path to image
        src = os.path.join('./TB data open source/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)

        image = cv2.imread(src)
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # save the image at the destination
        cv2.imwrite(dst, image)
        # shutil.copyfile(src, dst)

    if fname in folder_2:
        # source path to image
        src = os.path.join('./TB data open source/Montgomery/MontgomerySet/CXR_png', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)

        image = cv2.imread(src)
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # save the image at the destination
        cv2.imwrite(dst, image)

# In[22]:


# Transfer the val images

for image in val_list:

    fname = image
    label = df_data.loc[image, 'target']

    if fname in folder_1:
        # source path to image
        src = os.path.join('./TB data open source/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)

        image = cv2.imread(src)
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # save the image at the destination
        cv2.imwrite(dst, image)

        # copy the image from the source to the destination
        # shutil.copyfile(src, dst)
    if fname in folder_2:
        # source path to image
        src = os.path.join('./TB data open source/Montgomery/MontgomerySet/CXR_png', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)

        image = cv2.imread(src)
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # save the image at the destination
        cv2.imwrite(dst, image)

        # copy the image from the source to the destination
        # shutil.copyfile(src, dst)

# In[23]:


class_list = ['Normal', 'Tuberculosis']

for item in class_list:

    # We are creating temporary directories here because we delete these directories later.
    # create a base dir
    aug_dir = 'aug_dir'
    os.mkdir(aug_dir)
    # create a dir within the base dir to store images of the same class
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)

    # Choose a class
    img_class = item

    # list all images in that directory
    img_list = os.listdir('base_dir/train_dir/' + img_class)

    # Copy images from the class train dir to the img_dir e.g. class 'Normal'
    for fname in img_list:
        # source path to image
        src = os.path.join('base_dir/train_dir/' + img_class, fname)
        # destination path to image
        dst = os.path.join(img_dir, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    # point to a dir containing the images and not to the images themselves
    path = aug_dir
    save_path = 'base_dir/train_dir/' + img_class

    # Create a data generator
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(path,
                                              save_to_dir=save_path,
                                              save_format='png',
                                              target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                              batch_size=batch_size)

    # Generate the augmented images and add them to the training folders

    num_files = len(os.listdir(img_dir))

    # this creates a similar amount of images for each class
    num_batches = int(np.ceil((NUM_AUG_IMAGES_WANTED - num_files) / batch_size))

    # run the generator and create augmented images
    for i in range(0, num_batches):
        imgs, labels = next(aug_datagen)

    # delete temporary directory with the raw image files
    shutil.rmtree('aug_dir')



train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 10
val_batch_size = 10

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)



datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                      target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                      batch_size=val_batch_size,
                                      class_mode='categorical')

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(valid_path,
                                       target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                       batch_size=val_batch_size,
                                       class_mode='categorical',
                                       shuffle=False)



from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=1, verbose=1)

filepath = "transferlearning_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')



from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Dropout, GlobalAveragePooling2D
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.layers import Conv2D, BatchNormalization
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
# K.set_image_dim_ordering('th')
# K.common.image_dim_ordering('th')
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier




from keras.applications.inception_v3 import InceptionV3

# create the base pre-trained model
base_model = InceptionV3(weights=None, include_top=False, input_shape=(150, 150, 3))




from keras.applications.resnet50 import ResNet50

#base_model = ResNet50(weights=None, include_top=False, input_shape=(150, 150, 3))

# In[53]:


x = base_model.output
x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(2, activation='sigmoid')(x)




base_model.load_weights("/Users/fuqinwei/Desktop/summer program/inception_v3_weights.h5")

# In[54]:


#base_model.load_weights("/Users/fuqinwei/Downloads/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

# In[55]:


model = Model(inputs=base_model.input, outputs=predictions)

# In[56]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


print(model.summary())


batch_size = 10
epochs = 10




model.compile(Adam(lr=0.0001), loss='binary_crossentropy',
              metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2,
                              verbose=1, mode='max', min_lr=0.00001)

callbacks_list = [checkpoint, reduce_lr]

# history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
#                               validation_data=val_gen,
#                               validation_steps=val_steps,
#                               epochs=10, verbose=1,
#                               callbacks=callbacks_list)


#model.save('InceptionV3_3.0_weights.hdf5')
model.load_weights('InceptionV3_weights.hdf5')

scores = model.evaluate_generator(test_gen,steps=val_steps)

print("Testing Accuracy: %.2f%%" % (scores[1] * 100))

# In[85]:


# Get the labels of the test images.

test_labels = test_gen.classes


# make a prediction
predictions = model.predict_generator(test_gen, steps=val_steps, verbose=1)


# In[88]:


# argmax returns the index of the max value in a row
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

# In[71]:



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[80]:


# Define the labels of the class indices. These need to match the
# order shown above.
cm_plot_labels = ['Normal', 'Tuberculosis']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

# In[78]:


# Get the filenames, labels and associated predictions

# This outputs the sequence in which the generator processed the test images
test_filenames = test_gen.filenames

# Get the true labels
y_true = test_gen.classes

# Get the predicted labels
y_pred = predictions.argmax(axis=1)

# In[79]:
import keras
new = keras.metrics.binary_accuracy(y_true, y_pred, threshold=0.6)
print(new)

from sklearn.metrics import classification_report

# Generate a classification report

report = classification_report(y_true, y_pred, target_names=cm_plot_labels)

print(report)






