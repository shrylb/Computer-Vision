import imutils
import cv2
import numpy as np
import os

class ImageStitch:
    def image_stitch(self, images, lowe_ratio=0.75, max_Threshold=4.0, match_status=False, vertical=True):
        """
        Stitches two images together based on feature matching and homography.

        Args:
            images (tuple): Tuple of two images to be stitched together.
            lowe_ratio (float, optional): Ratio for feature matching. Defaults to 0.75.
            max_Threshold (float, optional): Maximum threshold for feature matching. Defaults to 4.0.
            match_status (bool, optional): Flag to indicate if match status should be returned. Defaults to False.
            vertical (bool, optional): Flag to indicate if the stitching is vertical. Defaults to True.

        Returns:
            Union[ndarray, Tuple[ndarray, ndarray]]: Stitched image or tuple of stitched image and match status.
        """
        # Extract images
        imageB, imageA = images

        # Detect features and keypoints
        key_points_A, features_of_A = self.detect_features(imageA)
        key_points_B, features_of_B = self.detect_features(imageB)

        # Match keypoints
        values = self.match_keypoints(key_points_A, key_points_B, features_of_A, features_of_B, lowe_ratio, max_Threshold)

        # If no matches found, return None
        if values is None:
            return None

        # Extract matches, homography, and status
        matches, homography, status = values

        # Get the warp perspective
        result_image = self.get_warp_perspective(imageA, imageB, homography, vertical)

        # Overlay the second image on the result image
        result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # Return the result image
        return result_image

    def get_warp_perspective(self, imageA, imageB, homography, vertical=True):
        # Calculate the dimensions of the resulting image by adding the heights and widths of the input images.
        height, width = (imageA.shape[0] + imageB.shape[0], max(imageA.shape[1], imageB.shape[1])) if vertical else (max(imageA.shape[0], imageB.shape[0]), imageA.shape[1] + imageB.shape[1])
        # Warp the perspective of imageA using the homography matrix and the calculated dimensions.
        # The resulting image will have the dimensions specified by the calculated height and width.
        return cv2.warpPerspective(imageA, homography, (width, height))

    def detect_features(self, image):
        """
        Detects SIFT (Scale-Invariant Feature Transform) features in the given image and computes their descriptors.

        Args:
            image (ndarray): Input image.

        Returns:
            Tuple[ndarray, ndarray]: Keypoints and features of the image.
        """
        # Create a SIFT detector.
        sift = cv2.SIFT_create()

        # Detect keypoints and compute their descriptors.
        # keypoints -  points of interest in the image.
        # features -  numerical descriptions of these points.
        keypoints, features = sift.detectAndCompute(image, None)

        # Return the keypoints and features as a tuple.
        return np.float32([kp.pt for kp in keypoints]), features

    def match_keypoints(self, keypointsA, keypointsB, featuresA, featuresB, lowe_ratio, max_Threshold):
        """
        Matches keypoints from two images using the Brute Force algorithm and RANSAC.

        Args:
            keypointsA (ndarray): Keypoints from the first image.
            keypointsB (ndarray): Keypoints from the second image.
            featuresA (ndarray): Features from the first image.
            featuresB (ndarray): Features from the second image.
            lowe_ratio (float): Lowe's ratio for removing outliers.
            max_Threshold (float): Maximum threshold for RANSAC.

        Returns:
            Tuple[List[Tuple[int, int]], ndarray, ndarray]: Valid matches, homography matrix, and status of inliers.
        """
        # Create a matcher object and find the nearest neighbors for each feature.
        # Finds the 2 best matches for each feature descriptor in the first image against those in the second image.
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        all_matches = matcher.knnMatch(featuresA, featuresB, 2)

        # Filter out matches using Lowe's test checks that the two distances are sufficiently different. 
        
        valid_matches = [(m[0].trainIdx, m[0].queryIdx) for m in all_matches if len(m) == 2 and m[0].distance < m[1].distance * lowe_ratio]

        # If there are less than 4 valid matches, return None.
        if len(valid_matches) <= 4:
            return None

        # Convert the keypoints to a float32 array.
        pointsA = np.float32([keypointsA[i] for (_, i) in valid_matches])
        pointsB = np.float32([keypointsB[i] for (i, _) in valid_matches])

        # Find the homography matrix using RANSAC and return the matches, homography matrix, and status of inliers.
        homography, status = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, max_Threshold)
        return valid_matches, homography, status

    def draw_matches(self, imageA, imageB, keypointsA, keypointsB, matches, status):
        """
        Draws the matches between two images.
        """
        # Combine the images side by side
        vis = self.get_combined_image(imageA, imageB)

        # Iterate over the matches and their corresponding status
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # If the match is valid (status = 1)
            if s == 1:
                # Get the coordinates of the keypoints
                ptA = (int(keypointsA[queryIdx][0]), int(keypointsA[queryIdx][1]))
                ptB = (int(keypointsB[trainIdx][0]) + imageA.shape[1], int(keypointsB[trainIdx][1]))

                # Draw a line between the two keypoints
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # Return the image with matches drawn
        return vis

    def get_combined_image(self, imageA, imageB):
        """
        Combines two images side by side.
        Returns:
            numpy.ndarray: The combined image.
        """
        # Calculate the height and width of the combined image.
        h = max(imageA.shape[0], imageB.shape[0])  # Maximum height of the two images.
        w = imageA.shape[1] + imageB.shape[1]  # Width of imageA + width of imageB.

        # Create a numpy array of zeros with the same height and width as the combined image.
        vis = np.zeros((h, w, 3), dtype="uint8")

        # Copy the pixels of imageA into the top-left part of the combined image.
        vis[0:imageA.shape[0], 0:imageA.shape[1]] = imageA

        # Copy the pixels of imageB into the top-right part of the combined image.
        vis[0:imageB.shape[0], imageA.shape[1]:] = imageB

        # Return the combined image.
        return vis
    

     # para sa pag remove sa black parts sa pag stitch vertically
    def crop_vertical(self, image):
        right_margin = 30
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 3: Apply a binary threshold to isolate the content
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Step 4: Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 5: Determine the bounding box of all contours combined
        if contours:
            all_contours = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_contours)
            
            # Adjust the width by reducing the right side
            new_w = max(1, w - right_margin)
            
            # Ensure width remains positive
            cropped_image = image[y:y+h, x:x+new_w]

            return cropped_image
        else:
            print("No contours found!")
            return image

    #para ma remove tong excess black parts sa overall image
    def crop_black_borders(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = image[y:y + h, x:x + w]
        return cropped

#Stitches vertically multiple images from the given data folder.
def vertical_stitching(data_folder):
    # Get the list of image files in the data folder
    image_files = sorted([f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))])

    # Divide the images into three groups
    left_images = [os.path.join(data_folder, image_files[i]) for i in [0, 3, 6]]
    mid_images = [os.path.join(data_folder, image_files[i]) for i in [1, 4, 7]]
    right_images = [os.path.join(data_folder, image_files[i]) for i in [2, 5, 8]]

    # Read and resize the images
    left_group = [cv2.imread(img) for img in left_images]
    mid_group = [cv2.imread(img) for img in mid_images]
    right_group = [cv2.imread(img) for img in right_images]

    for group in [left_group, mid_group, right_group]:
        for i in range(len(group)):
            group[i] = imutils.resize(group[i], width=400)

    # Create an instance of ImageStitch
    panorama = ImageStitch()
    stitched_images = []
    cropped_images = []  # List to hold cropped images

    # Stitch and crop the images
    for group, name in zip([left_group, mid_group, right_group], ["Left", "Mid", "Right"]):
        if len(group) == 2:
            result = panorama.image_stitch(group)
        else:
            result = panorama.image_stitch([group[-2], group[-1]])
            for img in reversed(group[:-2]):
                result = panorama.image_stitch([img, result])

        # Apply cropping only to the "Left" image
        if name == "Left":
            result = panorama.crop_vertical(result)
        # Apply cropping only to the "Right" image
        if name == "Right":
            result = panorama.crop_black_borders(result)

        cropped_images.append(result)  # Append (possibly cropped) image to the list
        stitched_images.append(result)  # Append stitched image to the list

    # Save the stitched and cropped images
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    vertical_result_paths = []
    for img, name in zip(stitched_images, ["Left.png", "Mid.png", "Right.png"]):
        output_path = os.path.join(output_folder, name)
        cv2.imwrite(output_path, img)  # Use img instead of result
        vertical_result_paths.append(output_path)

    return vertical_result_paths
    
    

def horizontal_stitching(image_paths):
    if len(image_paths) < 2:
        print("Not enough images for stitching.")
        return

    images = [cv2.imread(image_path) for image_path in image_paths]

    for i in range(len(images)):
        images[i] = imutils.resize(images[i], width=400)

    panorama = ImageStitch()
    if len(images) == 2:
        result = panorama.image_stitch([images[0], images[1]], match_status=False, vertical=False)
    else:
        result = panorama.image_stitch([images[-2], images[-1]], match_status=False, vertical=False)
        for img in reversed(images[:-2]):
            result = panorama.image_stitch([img, result], match_status=False, vertical=False)

    result = panorama.crop_black_borders(result)
    
    cv2.imwrite("output/Final_Stitched.png", result)
    cv2.imshow("Image Stitched", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# VERTICAL STITCHING 
data_folder = "data"
vertical_result_paths = vertical_stitching(data_folder)
# Load and display the Left.png, Mid.png, and Right.png images
left_img = cv2.imread("output/Left.png")
mid_img = cv2.imread("output/Mid.png")
right_img = cv2.imread("output/Right.png")

# Display the Left.png, Mid.png, and Right.png images
cv2.imshow("Left Image", left_img)
cv2.imshow("Mid Image", mid_img)
cv2.imshow("Right Image", right_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# HORIZONTAL STITCHING
horizontal_stitching(vertical_result_paths)