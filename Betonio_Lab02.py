import numpy as np
import cv2

def cross_correlation(img, kernel):
    # get diomensions
    image_height, image_width = img.shape[:2]
    kernel_height, kernel_width = kernel.shape
    
    # Pad the image
    image_padded = np.zeros((image_height + kernel_height - 1, image_width + kernel_width - 1))
    
    image_padded[int((kernel_height - 1) / 2):int((kernel_height - 1) / 2) + image_height,
                 int((kernel_width - 1) / 2):int((kernel_width - 1) / 2) + image_width] = img
    
    output = np.zeros_like(img)
    # Perform cross correlation
    for i in range(image_height):
        for j in range(image_width):
            window = image_padded[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(window * kernel)  # numpy does element-wise multiplication on arrays
    return output


def convolution(img, kernel):
   
    # For convolution, flip the kernel
    flipped_kernel = np.flipud(np.fliplr(kernel))
    #flipped_kernel = np.flip(kernel)
    # then use cross_correlation to perform convolution
    return cross_correlation(img, flipped_kernel)

def gaussian_blur(sigma, height, width):
    center=int(width/2)
    kernel=np.zeros((width,height))
    for i in range(width):
        for j in range(height):
              kernel[i,j] = (1/(2*np.pi*sigma**2))*np.exp(-((i-center)**2+(j-center)**2)/(2*sigma**2))
    kernel=kernel/np.sum(kernel)	#Normalize values so that sum is 1.0
    return kernel


    
def low_pass(img, sigma, size):
    # Create Gaussian blur kernel
    kernel = gaussian_blur(sigma, size, size)
    # Convolve the image with the kernel
    return convolution(img, kernel)

def high_pass(img, sigma, size):
    
    # Create low-pass filtered image
    low_pass_img = low_pass(img, sigma, size)
    # High-pass filter: original image - low-pass filtered image
    return img - low_pass_img

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)
        
    # mixing the images
    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    # scaling and clipping
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

# RUNNING THE FUNCTIONS
# pix2 -> einstein, pix1 -> monroe
img1 = cv2.imread('pix1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('pix2.jpg', cv2.IMREAD_GRAYSCALE)

# Resize img2 to match img1's dimensions
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

#Defining parameters
sigma1 = 10     # the smaller ang number mas makita nako si monroe
size1 = 30     # the bigger the number, mas maklaro na ang edges ni einstein, and smoother na si monroe
high_low1 = 'low'
sigma2 = 9
size2 = 10
high_low2 = 'high'
mixin_ratio = 0.5
scale_factor = 1.0

#Create hybrind image
hybrid_img = create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio, scale_factor)

cv2.imshow('Left Images', img1)
cv2.imshow('Right Image', img2)
cv2.namedWindow('Hybrid Image', cv2.WINDOW_NORMAL)
cv2.imshow('Hybrid Image', hybrid_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

