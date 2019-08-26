#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('mnist_train.csv')
print(df.shape)


# In[78]:


df.head(5)


# In[5]:


df.describe()


# In[6]:


data = df.values


# In[7]:


X = data[:,1:]
Y = data[:,0]
print(X.shape)
print(Y.shape)


# In[8]:


from sklearn.cross_validation import train_test_split


# In[9]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# #### Can we use KNN?
# ![image.png](attachment:image.png)

# In[10]:


def dist(a,b):
    return np.sqrt(sum((a-b)**2))

def knn(X,Y,queryPoint,k=5):
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        d = dist(X[i],queryPoint)
        vals.append([d,Y[i]])
        
    vals = sorted(vals)
    vals = vals[:k]
    vals = np.array(vals)
    counts = np.unique(vals[:,1],return_counts=True)
    max_freq_index = counts[1].argmax()
    prediction = counts[0][max_freq_index]
    return prediction


# In[174]:


for i in range(Y_train.shape[0]):
    if Y_train[i]==5:
        print(i)


# In[244]:


print(Y_train[3])
print(X_train[3])


# In[175]:


# Visualize some samples

def drawImg(sample):
    img =  sample.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.show()

drawImg(X_train[3])
print(Y_train[19])


# In[159]:


pred = knn(X_train,Y_train,X_test[0])
print(int(pred))
print(Y_test[0])


# In[160]:


X_test[0].shape


# In[ ]:


#Method to check accuracy.
correct = 0
m = X_test.shape[0]
for i in range(m):
    pred = knn(X_train,Y_train,X_test[i])
    if int(pred)==Y_test[i]:
        correct = correct+1
accuracy = (correct/X_test.shape[0])*100
print("accuracy is ",accuracy,"%")


# ## My try for recognizing handwritten images

# In[176]:


import cv2


# In[234]:


handwritten_img = cv2.imread('handwritten.jpg',cv2.IMREAD_GRAYSCALE)
crop_image = handwritten_img[105:520,350:650]
final_handwritten_image = cv2.resize(crop_image,(28,28))


# In[235]:


plt.imshow(handwritten_img)


# In[236]:


plt.imshow(crop_image)


# In[237]:


plt.imshow(final_handwritten_image,cmap='gray')


# In[238]:


print(final_handwritten_image.shape)


# In[ ]:





# In[239]:


#final_handwritten_image


# In[240]:


final_handwritten_image = final_handwritten_image.reshape(784,)
#final_handwritten_image_optimized = 125 - final_handwritten_image
print(final_handwritten_image.min())
print(final_handwritten_image.max())


# In[246]:


for i in range(final_handwritten_image.shape[0]):
    if final_handwritten_image[i]>90:
        final_handwritten_image_optimized[i]=0
    else:
        final_handwritten_image_optimized[i]=(300-final_handwritten_image[i])


# In[247]:


print(final_handwritten_image_optimized)


# In[248]:


plt.imshow(final_handwritten_image_optimized.reshape(28,28),cmap='gray')


# In[249]:


pred = knn(X_train,Y_train,final_handwritten_image_optimized)
print(int(pred))


# In[ ]:




