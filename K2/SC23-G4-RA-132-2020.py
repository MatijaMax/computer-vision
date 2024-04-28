import os
import numpy as np
import cv2 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import csv
import sys


#FUNCTIONS and DEFINITIONS

class Data:
    def __init__(self, video, solution):
        self.video = video
        self.solution = solution

def load_image(path):
    return cv2.imread(path)

def classify_window(window):
    features = hog.compute(window).reshape(1, -1)
    return clf_svm.predict_proba(features)[0][1]

def detect_cross_ultra(y):    
    return -1 <= (1080 - y) <= 1

def process_video(video_path):
    sum_of_cars = 0

    
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num) # indexing frames
    
    while True:
        frame_num += 1
        grabbed, frame = cap.read()

        if not grabbed:
            break
             
        rectangles = process_images(frame, 60)
        #print(len(rectangles))
        for rectangle in rectangles:
            x1, y1, x2, y2 = rectangle 
            #print(y1)
            #center
            center_y = y1 + 120 / 2
                
            if (detect_cross_ultra(center_y)):    
                sum_of_cars += 1
                #print(f"The sum is {sum_of_cars}")

    cap.release()
    return sum_of_cars

def process_images(image, step_size, window_size=(120, 240), resize_to=(60, 120), min_score=0.7): # 0.7(~1.5) >> 0.9(~3.75); 0.5(~3.75); 0.6(~2.25)   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    all_windows = []
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            window = image[y:y+window_size[1], x:x+window_size[0]]  
            if window.shape == (window_size[1], window_size[0]):
                resized_window = cv2.resize(window, (resize_to[0], resize_to[1]))
                score = 0
                #if((1640<x<2240)):
                #Equal MAE but 3 times faster                    
                if((1640<x<2240) and (880<y<1280)): 
                    score = classify_window(resized_window)
                if score > min_score:
                    all_windows.append((x, y, x+resize_to[0], y+resize_to[1]))
    filtered_windows = non_max_suppression_fast(np.array(all_windows), overlapThresh=0.5) ### MODIFY  
    #print(len(filtered_windows))              
    return filtered_windows

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = x1 +  60
	y2 = y1 +  120
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

#############
#   START   #
#############

### Load training set
train_dir = sys.argv[1]+'pictures'

pos_imgs = []
neg_imgs = []

for img_name in os.listdir(train_dir):
    img_path = os.path.join(train_dir, img_name)
    img = load_image(img_path)
    if 'p_' in img_name:
        pos_imgs.append(img)
    elif 'n_' in img_name:
        neg_imgs.append(img)
        
#print("Positive images #: ", len(pos_imgs))
#print("Negative images #: ", len(neg_imgs))

### Calculate HOG descriptor (1-car 0-other)

pos_features = []
neg_features = []
labels = []

nbins = 9 # number of bins
cell_size = (8, 8) # number of pixels per cell
block_size = (3, 3) # number of cells per block


hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)
                     

for img in pos_imgs:
    pos_features.append(hog.compute(img))
    labels.append(1)

for img in neg_imgs:
    neg_features.append(hog.compute(img))
    labels.append(0)

pos_features = np.array(pos_features)
neg_features = np.array(neg_features)
x = np.vstack((pos_features, neg_features))
y = np.array(labels)

#3 Set division (train and validation sets)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#print('Train shape: ', x_train.shape, y_train.shape)
#print('Test shape: ', x_test.shape, y_test.shape)

#4 Training SVM classifier

clf_svm = SVC( kernel='linear', probability=True) 
clf_svm.fit(x_train, y_train)
y_train_pred = clf_svm.predict(x_train)
y_test_pred = clf_svm.predict(x_test)
#print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
#print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

#5 Processing video

#count = process_video("data1/videos/segment_1.mp4")
#print("Calculated sum: ", count)

#6 FINAL MAE

data_list=[]
with open((sys.argv[1]+'counts.csv'), 'r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)
    
    # Skip the first row (header)
    next(csv_reader)

    # Iterate through the remaining rows
    for row in csv_reader:
        video, solution = row
        data_instance = Data(video, int(solution))
        data_list.append(data_instance)

sum_error = 0
for data in data_list:
    prediction=process_video(sys.argv[1]+'videos/'+data.video)
    sum_error += np.abs(data.solution - prediction)
    print(f"{data.video}-{data.solution}-{prediction}")

print(f"MAE: {1/4*sum_error}")


