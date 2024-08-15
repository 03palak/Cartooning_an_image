#!/usr/bin/env python
# coding: utf-8

# In[27]:


pip install opencv-python


# In[28]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[29]:


def read_file(filename):
   img = cv2.imread(filename)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   plt.imshow(img) 
   plt.show()
   return img


# In[30]:


filename ="Image.jpg"
img = read_file(filename)

org_img = np.copy(img)


# CREATE EDGE MASK

# In[31]:


def edge_mask(img, line_size, blur_value):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur
    gray_blur = cv2.medianBlur(gray, blur_value)
    
    # Apply adaptive threshold
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    
    return edges


# In[32]:


line_size, blur_value = 7, 7
edges = edge_mask(img, line_size, blur_value)

plt.imshow(edges , cmap="gray")
plt.show()


# In[33]:


import cv2

def color_quantization(img, k):
    # Reshape the image to a 2D array of pixels
    pixels = img.reshape((-1, 3))
    
    # Convert pixels to float32
    pixels = np.float32(pixels)
    
    # Determine Criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing k-means
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to uint8
    centers = np.uint8(centers)
    
    # Map the labels to the centers
    segmented_img = centers[labels.flatten()]
    
    # Reshape back to the original image shape
    segmented_img = segmented_img.reshape(img.shape)
    
    return segmented_img

# Example usage


# In[34]:


img = color_quantization(img, k=9)
plt.imshow(img)
plt.show()


# In[35]:


#Reduce the noise
blurred = cv2.bilateralFilter(img, d=2,sigmaColor = 200,sigmaSpace=200)

plt.imshow(blurred)
plt.show()


# #combine edge mask with quantize image

# In[36]:


def cartoon():
    c=cv2.bitwise_and(blurred, blurred,mask=edges)
    
    plt.imshow(c)
    plt.title("Cartoonified Image")
    plt.show()
    
    plt.imshow(org_img)
    plt.title("org_img")
    plt.show()


# In[37]:


cartoon()


# In[ ]:





# In[ ]:





# In[ ]:




