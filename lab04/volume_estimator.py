import os
import numpy as np
import cv2

# Calibration constants
THRESHOLD = 90                 # Threshold value for binary thresholding
PIXEL_VALUE = 750              # Value to convert pixel count to volume
MIN_AREA_SIZE = 1000            # Minimum contour area size to consider
MAX_AREA_SIZE = 350000          # Maximum contour area size to consider


def display_options():
    '''
    Displays options for selecting the image set and returns the selected option.
    '''
    os.system('cls')
    print("This program estimates the volume of liquid in a container using image processing.\n")
    print("SELECT IMAGE SET")
    print("1. 50mL\t\t4. Unknown A")
    print("2. 150mL\t5. Unknown B")
    print("3. 250mL\t6. Unknown C")
    return input("\nPlease select an option: ")

def process_image(image):
    '''
    Processes the image to estimate the volume of liquid.
    - Converts the image to grayscale
    - Applies thresholding, dilation, and erosion
    - Finds and filters contours
    - Calculates the volume based on the largest contour area
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    binary = cv2.dilate(binary, None, iterations= 7 )

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cv2.contourArea(c) for c in contours if MIN_AREA_SIZE < cv2.contourArea(c) < MAX_AREA_SIZE]
    
    if not filtered_contours:
        return 0.0

    largest_area = max(filtered_contours)
    return largest_area / PIXEL_VALUE

def calculate_average_volume(directory):
    '''
    Calculates the average volume of liquid for all images in the given directory.
    '''
    total_volume = 0
    files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))]
    
    if not files:
        print(f"No image files found in directory: {directory}")
        return

    for file in files:
        image_path = os.path.join(directory, file)
        image = cv2.imread(image_path)
        volume = process_image(image)
        print(f"Processing image: {file} -> Estimated volume: {volume:.2f} mL")
        total_volume += volume

    average_volume = total_volume / len(files)
    print(f"\nAverage estimated volume: {average_volume:.2f} mL")

def main():
    selected_option = display_options()
    directory_map = {
        "1": "Knowns/50mL",
        "2": "Knowns/150mL",
        "3": "Knowns/250mL",
        "4": "guess/A",
        "5": "guess/B",
        "6": "guess/C"
    }

    if selected_option not in directory_map:
        print("Invalid selection.")
        return

    directory = directory_map[selected_option]
    calculate_average_volume(directory)

if __name__ == "__main__":
    main()
