import numpy as np
import random
import math
import cv2
from frequency_filtering import dft
from numpy.random import rand
import numpy as np

input_img=cv2.imread('Lenna.png',0)
img_h, img_w=input_img.shape[0:2]
output=np.zeros_like(input_img)

def gauskernel(n, sigma): # Note, 'n'*2+1 = kernel shape, sigma is the standard deviation
    x = np.arange(-n, n+1, 1)
    y = np.arange(-n, n+1, 1)
    x2d, y2d = np.meshgrid(x, y)
    kernel = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    return kernel / (2 * np.pi * sigma ** 2) # unit integral

kernelz = gauskernel(n = 2, sigma = 1)

G=np.zeros((5,5))
sigma=1
print(input_img.shape[:])
for x in range(5):
    for y in range(5):
        G[x,y]= (1/(2*np.pi*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))
normalized_f=sum(G)
G=(1/normalized_f)*G
kernel=kernelz

kernel_h, kernel_w = kernel.shape[0:2]
# Add zero padding to the input image
image_padded = np.zeros((img_h + (kernel_h - 1),
                         img_w + (kernel_w - 1)))
print(image_padded.shape[:])


image_padded[(kernel_h // 2):-(kernel_h // 2),
(kernel_w // 2):-(kernel_w // 2)] = input_img
image_padded[(kernel_h // 2):-(kernel_h // 2),
(kernel_w // 2):-(kernel_w // 2)]
for x in range(img_w):  # Loop over every pixel of the image
    for y in range(img_h):
        # element-wise multiplication of the kernel and the image

        output[y, x] = (kernel * image_padded[y:y + kernel_h, x:x + kernel_w]).sum()
#cv2.imshow('yes',output)
#cv2.waitKey(0)
p=np.matrix(([1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25],[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]))
print(p)
print(p[2:-2,2:-2])


kernel=np.matrix([[0,1,0],[1,-4,1],[0,1,0]])
#kernel=np.matrix([[1,1,1],[1,-8,1],[1,1,1]])
#kernel=np.matrix([[-1,2,-1],[2,-4,2],[-1,2,-1]])
kernel_h, kernel_w = kernel.shape[0:2]
# Add zero padding to the input image
image_padded = np.zeros((img_h + (kernel_h - 1),
                         img_w + (kernel_w - 1)))
print(image_padded.shape[:])


image_padded[(kernel_h // 2):-(kernel_h // 2),
(kernel_w // 2):-(kernel_w // 2)] = input_img
image_padded[(kernel_h // 2):-(kernel_h // 2),
(kernel_w // 2):-(kernel_w // 2)]
for x in range(img_w):  # Loop over every pixel of the image
    for y in range(img_h):
        # element-wise multiplication of the kernel and the image

        output[y, x] = (kernel * image_padded[y:y + kernel_h, x:x + kernel_w]).sum()
        if output[y,x]<0:
            output[y,x]=0
        if output[y,x]>255:
            output[y,x]=255


img=input_img
h,w=input_img.shape[:2]
processed_img = np.zeros((h + 2, w + 2), 'uint8')

new_img = np.zeros((h + 2, w + 2), 'uint8')
r_c = 0
for r in img:
    new_img[r_c + 1][1:-1] = r
    r_c += 1
for u in range(0, h):
    for v in range(0, w):
        new_pxl = 0
        for i in range(3):
            for j in range(3):
                new_pxl += (new_img[u + i][v + j] * kernel[i][j])
        if new_pxl < 0:
            new_pxl = 0
        if new_pxl > 255:
            new_pxl = 255
        processed_img[u][v] = new_pxl
cv2.imshow('i',processed_img)
cv2.waitkey(0)

def get_laplacian_filter():
    """Initialzes and returns a 3X3 Laplacian filter"""
    laplacian_filter = np.zeros((3, 3))
    for u in range(0, 3):
        for v in range(0, 3):
            if u in [0, 2] and v in [0, 2]:  # 4 corner
                laplacian_filter[u][v] = 0
            elif (u == 1 and v in [0, 2]) or (v == 1 and u in [0, 2]):  # 4 sides
                laplacian_filter[u][v] = 1
            elif u == 1 and v == 1:  # center
                laplacian_filter[u][v] = -4
    return laplacian_filter
print(get_laplacian_filter())