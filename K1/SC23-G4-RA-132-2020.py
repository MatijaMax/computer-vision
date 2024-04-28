#imports
import numpy as np
import cv2 # OpenCV
import matplotlib
from matplotlib import pyplot as plt
import csv
import sys

#image size def
matplotlib.rcParams['figure.figsize'] = 14,18

#data class
class Data:
    def __init__(self, image, solution):
        self.image = image
        self.solution = solution

#OpenCV image manipulation functions
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return image_bin

def image_bin_adapt(image_gs):
    #image_bin =cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 95, 10)
    #image_bin =cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 7)
    image_bin =cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 555, 1)
    return image_bin

def invert(image):
    return 255-image

def predict_value(image):
    #image = load_image("pictures1\picture_1.jpg") 
    gray = image_gray(image)
    #gray=image_bin(gray)
    #plt.imshow(gray, cmap='gray')
    #plt.show()
    #blur = cv2.GaussianBlur(gray, (1, 1), 135)
    #plt.imshow(blur, cmap='gray')
    #plt.show()
    canny = cv2.Canny(gray, 30, 70, 3) # edge detection 
    #plt.imshow(canny, cmap='gray')
    #plt.show() 
    dilated = cv2.dilate(canny, (5, 5), iterations=4) # edge thickening
    dilated=invert(dilated) 
    #plt.imshow(dilated, cmap='gray')
    #plt.show() 
    cnt, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
    cv2.drawContours(image, cnt, -1, (255, 0, 0), 1)
    #plt.imshow(image)
    #plt.show()
    contours_pokemon=[]
    for contour in cnt: 
        if len(contour) >= 5:  # At least 5 points are required for fitting an ellipse !!!!!
            ellipse = cv2.fitEllipse(contour)
            center, axis, angle = ellipse  # Get ellipse parameters
            major_axis, minor_axis = axis

            #drawing ellipses for pokemon contours
            if major_axis > 30 and major_axis < 100 and minor_axis > 20 and minor_axis < 100 and (angle < 30 or angle > 155):
                contours_pokemon.append(contour)  # This contour belongs to the barcode
                cv2.ellipse(image, ellipse, (255, 0, 0), 2)

    #print('Number of pokemon is :', len(contours_pokemon))
    #plt.imshow(image) 
    #plt.show()
    return len(contours_pokemon)


data_list=[]
###init data
'''
data = Data('picture_1.jpg', 4) 
data_list.append(data)
data = Data('picture_2.jpg', 8)  
data_list.append(data)
data = Data('picture_3.jpg', 6)  
data_list.append(data)
data = Data('picture_4.jpg', 8)  
data_list.append(data)
data = Data('picture_5.jpg', 8)  
data_list.append(data)
data = Data('picture_6.jpg', 4)  
data_list.append(data)
data = Data('picture_7.jpg', 6)  
data_list.append(data)
data = Data('picture_8.jpg', 6)  
data_list.append(data)
data = Data('picture_9.jpg', 6)  
data_list.append(data)
data = Data('picture_10.jpg', 13)  
data_list.append(data)
'''

with open('squirtle_count.csv', 'r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)
    
    # Skip the first row (header)
    next(csv_reader)

    # Iterate through the remaining rows
    for row in csv_reader:
        image, solution = row
        # Create a Data instance for each row
        data_instance = Data(image, int(solution))
        data_list.append(data_instance)
        # You can now work with the data_instance
        # print(f"Image: {data_instance.image}, Solution: {data_instance.solution}")



#for data in data_list:
    #print(f"Image: {data.image}, Solution: {data.solution}")
sum_error = 0
for data in data_list:
    #print(sys.argv[1])
    image=load_image(sys.argv[1]+data.image)
    prediction=predict_value(image)
    sum_error += np.abs(data.solution - prediction)
    print(f"{data.image}-{data.solution}-{prediction}")

print(f"MAE: {1/10*sum_error}")