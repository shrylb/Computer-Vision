import os
import cv2
import numpy as np

# Load Image
image_path = 'TestImage2.jpg'
#image_path = 'MyPic.png'

image = cv2.imread(image_path)
if image is None:
    print("Error: Unable to load image.")
    exit()
    
# Getting dimensions
height1,width1, channel1 = image.shape

# Calculate the width of each strip
num_strips = 100
strip_width1 = width1 // num_strips

# Split the image vertically into strips
strips1 = []
for i in range(num_strips):
    start_x = i * strip_width1
    end_x = min((i + 1) * strip_width1, width1)  # Adjust the end position for the last strip using min
    strip1 = image[:, start_x:end_x]
    strips1.append(strip1)
    
# Separate even and odd strips
even_strips1 =[]
odd_strips1 = []
for i, strip in enumerate(strips1):
    if i % 2 == 0:
        even_strips1.append(strip)
    else:
        odd_strips1.append(strip)

# Combining even and odd strips into one image
# similar to concatenating them horizontally
#   -----------TRIAL---------------------------

def concatenate_images(images):
    total_width = sum(image.shape[1] for image in images)
    max_height = max(image.shape[0] for image in images)
    concatenated_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    x_offset = 0
    for image in images:
        height, width, _ = image.shape
        concatenated_image[:height, x_offset:x_offset+width] = image
        x_offset += width
    return concatenated_image


combined_even1 = concatenate_images(even_strips1)
combined_odd1 = concatenate_images(odd_strips1)

tocombine1 = [combined_even1, combined_odd1]
combined1 = concatenate_images(tocombine1)

# ROTATING THE IMAGE
rotate_direction1 = cv2.ROTATE_90_CLOCKWISE
rotated_image = cv2.rotate(combined1, rotate_direction1)

height2,width2, channels2 = rotated_image.shape
strip_width2 = width2 // num_strips

strips2 = []
for i in range(num_strips):
    start_x = i * strip_width2                  # get starting x-coordinate
    end_x = min((i + 1) * strip_width2, width2)  # get ending x-coordinate
    strip2 = rotated_image[:, start_x:end_x]    # extract strip
    strips2.append(strip2)                      # add strip sa list
    
even_strips2 =[]
odd_strips2 = []
for i, strip in enumerate(strips2):
    if i % 2 == 0:
        even_strips2.append(strip)
    else:
        odd_strips2.append(strip)

# concatenate after stripping it 
combined_even2 = concatenate_images(even_strips2)
combined_odd2 = concatenate_images(odd_strips2)

tocombine2 = [combined_even2, combined_odd2]
combined2 = concatenate_images(tocombine2)

# then rotate image counterclckwise
rotate_direction2 = cv2.ROTATE_90_COUNTERCLOCKWISE
final_image = cv2.rotate(combined2, rotate_direction2)

# turn image to grey
last_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
resized_image = cv2.resize(last_image, (0, 0), fx=2, fy=2)

cv2.imshow("Output", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

    
