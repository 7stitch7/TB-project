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
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


PathDicom = "/Users/fuqinwei/Desktop/summer program/PHRU Data/image/JPEG"
lstFilesDCMA = []  # create an empty list
lstFilesDCMB = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's jpg
            if "-1" not in filename.lower():
                if "b" in filename.lower():
                    lstFilesDCMB.append(os.path.join(dirName,filename))
                else:
                    lstFilesDCMA.append(os.path.join(dirName,filename))


# In[4]:


lstFilesDCMA.sort() 
lstFilesDCMB.sort() 


# In[5]:


label = pd.read_csv('/Users/fuqinwei/Desktop/summer program/PHRU Data/label.csv')


# In[6]:


df_label = label.replace({'Abnormal':1,'Normal in both lungs':0})


# In[7]:


df_label.head()


# In[8]:


# Total number of images we want to have in each class
NUM_AUG_IMAGES_WANTED = 1500 

# We will resize the images
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150


# In[9]:


img_id = []
for i in range(len(lstFilesDCMA)):
    img_id.append(lstFilesDCMA[i])
for i in range(len(lstFilesDCMB)):
    img_id.append(lstFilesDCMB[i])


# In[10]:


len(img_id)


# In[11]:


df_img=pd.DataFrame(data=img_id, columns=['imgID']) 


# In[12]:


laA = np.array(df_label.findings_cxr)
laB = np.array(df_label.findings_cxr_56)
la = np.append(laA,laB)


# In[13]:


df_la = pd.DataFrame(la,columns = ['label'])


# In[14]:


df_data = pd.concat([df_img, df_la], axis=1).reset_index(drop=True)
df_data.shape


# In[15]:


pathological_label = pd.read_csv('/Users/fuqinwei/Desktop/summer program/PHRU Data/pathological_label.csv')


# In[16]:


pathological_label.head()


# In[17]:


df_Plabel = pathological_label.replace({'Yes':1,'No':0,None:0})


# In[18]:


df_Plabel['cavity']=df_Plabel['cavity_ll_cxr']+df_Plabel['cavity_rl_cxr']
df_Plabel['infiltrates']=df_Plabel['infiltrates_ll_cxr']+df_Plabel['infiltrates_rl_cxr']
df_Plabel['adenopathy']=df_Plabel['adenopathy_ll_cxr']+df_Plabel['adenopathy_rl_cxr']
df_Plabel['pe']=df_Plabel['pe_ll_cxr']+df_Plabel['pe_rl_cxr']


# In[19]:


df_Plabel['cavity_56']=df_Plabel['cavity_ll_cxr_56']+df_Plabel['cavity_rl_cxr_56']
df_Plabel['infiltrates_56']=df_Plabel['infiltrates_ll_cxr_56']+df_Plabel['infiltrates_rl_cxr_56']
df_Plabel['adenopathy_56']=df_Plabel['adenopathy_ll_cxr_56']+df_Plabel['adenopathy_rl_cxr_56']
df_Plabel['pe_56']=df_Plabel['pe_ll_cxr_56']+df_Plabel['pe_rl_cxr_56']


# In[20]:


df_la['cavity'] = pd.concat([df_Plabel.cavity, df_Plabel.cavity_56], axis=0).reset_index(drop=True)
df_la['infiltrates'] = pd.concat([df_Plabel.infiltrates, df_Plabel.infiltrates_56], axis=0).reset_index(drop=True)
df_la['adenopathy'] = pd.concat([df_Plabel.adenopathy, df_Plabel.adenopathy_56], axis=0).reset_index(drop=True)
df_la['pe'] = pd.concat([df_Plabel.pe, df_Plabel.pe_56], axis=0).reset_index(drop=True)


# In[21]:


Dataset =pd.concat([df_data,df_la],axis=1) 


# In[22]:


Dataset = shuffle(Dataset)


# In[23]:


data = Dataset.replace(2,1)


# In[ ]:





# In[24]:


data = data.T.drop_duplicates().T


# In[ ]:





# In[25]:


data = data[data['label']==1]


# In[26]:


data


# In[27]:


path = '/Users/fuqinwei/TB project/TB/base_dir/val_dir/Normal'
lstFiles = []  # create an empty list
for dirName, subdirList, fileList in os.walk(path):
    for filename in fileList:
        if ".png" in filename.lower():  # check whether the file's jpg
            lstFiles.append(os.path.join(dirName,filename))


# In[28]:


normal = []
for i in lstFiles:
    img = cv2.imread(i)
    img = cv2.resize(img,(150,150))
    normal.append(img)


# In[29]:


normal_la=np.zeros((61,),dtype=int)


# In[28]:


y = ['cavity','infiltrates','adenopathy','pe']
label = []
for i in y:
    label.append(np.array(data[i]))


# In[31]:


label[0]=np.append(label[0],normal_la)
label[1]=np.append(label[1],normal_la)
label[2]=np.append(label[2],normal_la)
label[3]=np.append(label[3],normal_la)


# In[ ]:





# In[29]:


imgdata = []
for i in data.imgID:
    imgData = cv2.imread(i)
    imgData = cv2.resize(imgData, (150, 150))
    imgdata.append(imgData)
# for i in lstFiles:
#     img = cv2.imread(i)
#     img = cv2.resize(img,(150,150))
#     imgdata.append(img)


# In[33]:


imagedata = np.array(imgdata)
permutation = np.random.permutation(len(label[0]))
shuffled_dataset = imagedata[permutation, :, :]
label[0] = label[0][permutation]
label[1] = label[1][permutation]
label[2] = label[2][permutation]
label[3] = label[3][permutation]


# In[30]:


train_num = 120
val_num = 30


# In[32]:


x_train = np.array(imgdata[:train_num])/255
x_val = np.array(imgdata[-val_num:])/255


# In[33]:


y_train = [label[0][:train_num],label[1][:train_num],label[2][:train_num],label[3][:train_num]]
y_val = [label[0][-val_num:],label[1][-val_num:],label[2][-val_num:],label[3][-val_num:]]


# # build model

# In[34]:


from keras.utils.np_utils import to_categorical
train_la = []
for i in y_train:
    train_la.append(to_categorical(i, 3))
va_la = []
for i in y_val:
    va_la.append(to_categorical(i, 3))


# In[35]:


from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint , LearningRateScheduler
lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=0.0001, patience=1, verbose=1)


# In[36]:


filepath="MHM_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_output2_accuracy', verbose=1, save_best_only=True, mode='max')


# In[37]:


from keras.models import Sequential , Model
from keras.layers import Dense , Activation
from keras.layers import Dropout , GlobalAveragePooling2D
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD , RMSprop , Adadelta , Adam
from keras.layers import Conv2D , BatchNormalization
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


# In[38]:


from keras.applications.inception_v3 import InceptionV3
# create the base pre-trained model
base_model = InceptionV3(weights=None, include_top=False , input_shape=(150, 150, 3))


# In[39]:


base_model.load_weights("/Users/fuqinwei/Desktop/summer program/inception_v3_weights.h5")


# In[40]:


x = base_model.output
x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(2, activation='sigmoid')(x)


# In[41]:


model = Model(inputs=base_model.input, outputs=predictions)


# In[42]:


model.load_weights('/Users/fuqinwei/Desktop/TB/InceptionV3_weights.hdf5')


# In[43]:


model.layers.pop()
model.layers.pop()


# In[44]:


model.layers.pop()


# In[45]:


exten1=Dense(1024, activation='relu')(model.output)
exten1 = BatchNormalization()(exten1)
exten1=Dropout(0.5)(exten1)
exten1=Dense(512, activation='relu')(exten1)
exten1 = BatchNormalization()(exten1)
exten1=Dropout(0.5)(exten1)
exten1=Dense(256, activation='relu')(exten1)
exten1=Dropout(0.5)(exten1)
exten1=Dense(3, activation='softmax',name='output1')(exten1)

exten2=Dense(1024, activation='relu')(model.output)
exten2 = BatchNormalization()(exten2)
exten2=Dropout(0.5)(exten2)
exten2=Dense(512, activation='relu')(exten2)
exten2 = BatchNormalization()(exten2)
exten2=Dropout(0.5)(exten2)
exten2=Dense(256, activation='relu')(exten2)
exten2=Dropout(0.5)(exten2)
exten2=Dense(3, activation='softmax',name='output2')(exten2)

exten3=Dense(1024, activation='relu')(model.output)
exten3 = BatchNormalization()(exten3)
exten3=Dropout(0.5)(exten3)
exten3=Dense(512, activation='relu')(exten3)
exten3 = BatchNormalization()(exten3)
exten3=Dropout(0.5)(exten3)
exten3=Dense(256, activation='relu')(exten3)
exten3=Dropout(0.5)(exten3)
exten3=Dense(3, activation='softmax',name='output3')(exten3)

exten4=Dense(1024, activation='relu')(model.output)
exten4 = BatchNormalization()(exten4)
exten4=Dropout(0.5)(exten4)
exten4=Dense(512, activation='relu')(exten4)
exten4 = BatchNormalization()(exten4)
exten4=Dropout(0.5)(exten4)
exten4=Dense(256, activation='relu')(exten4)
exten4=Dropout(0.5)(exten4)
exten4=Dense(3, activation='softmax',name='output4')(exten4)

model = Model(inputs=[base_model.input], outputs=[exten1, exten2,exten3,exten4])


# In[46]:


print(model.summary())


# In[47]:


batch_size = 64
epochs = 10


# In[48]:


model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', 
              metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_output1_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
# hist = model.fit_generator(datagen.flow(x_train, [np.array(Pn_y_train),np.array(Tb_y_train)],
#                                  batch_size=batch_size),
#                     steps_per_epoch=x_train.shape[0] // batch_size,
#                     validation_data=(x_test,[np.array(Pn_y_test),np.array(Tb_y_test)]),epochs=10
#                            ,verbose=1,callbacks=[lr_reduce,checkpoint])
model.fit(np.array(x_train), train_la ,batch_size=10,verbose=1 ,callbacks=[lr_reduce,checkpoint],validation_data=(np.array(x_val),va_la),epochs=10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




