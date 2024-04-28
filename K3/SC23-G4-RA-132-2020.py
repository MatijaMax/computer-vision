import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.cluster import KMeans
import csv
import sys


class Data:
    def __init__(self, image, solution):
        self.image = image
        self.solution = solution

def hamming_distance(str1, str2):
    length = min(len(str1), len(str2))    
    distance = sum(ch1 != ch2 for ch1, ch2 in zip(str1[:length], str2[:length]))  
    distance += abs(len(str1) - len(str2))   
    return distance

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def erode(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=4)

def erode_two(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=2)

def erode_three(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=6)

def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

def select_roi_with_distances(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 100 and h < 100 and h > 35 and w > 20   and x >= 225 and x + w <= 875 and y >=100 and y + h <= 425:
            region = image_bin[y:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    x_values = [region[1][0] for region in regions_array]
    width_values = [region[1][2] for region in regions_array]
    region_distances = []
    # izdvojiti sortirane parametre opisujucih pravougaonika
    # izracunati rastojanja izmedju svih susednih regiona po X osi i dodati ih u niz rastojanja
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2]) # x_next - (x_current + w_current)
        region_distances.append(distance)
    
    return image_orig, sorted_regions, region_distances, x_values, width_values

def display_result_with_spaces(outputs, alphabet, k_means):
    # odredjivanje indeksa grupe koja odgovara rastojanju izmedju reci
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    # iterativno dodavanje prepoznatih elemenata
    # dodavanje space karaktera ako je rastojanje izmedju dva slova odgovara rastojanju izmedju reci
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]
    return result

def display_result(outputs, alphabet):
    result = ''
    for output in outputs:
        result += alphabet[winner(output)]
    return result

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann

def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32) # dati ulaz
    y_train = np.array(y_train, np.float32) # zeljeni izlazi na date ulaze
    
    print("\nTraining started...")
    sgd = SGD(learning_rate=0.1, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    print("\nTraining completed...")
    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def erode_test(image, index):
    img_bin = image
    image_orig, sorted_regions, region_distances, x_values, w_values = select_roi_with_distances(img, img_bin)
    print(region_distances)
    #cv2.imshow('Modified Image with Marked Regions', image_orig)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    region_of_interest = img_bin[:, x_values[index]:x_values[index]+w_values[index]]
    eroded_region = erode_three(region_of_interest)
    img_bin[:, x_values[index]:x_values[index]+w_values[index]] = eroded_region
    #cv2.imshow('Modified Image', img_bin)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    image_orig, sorted_regions, region_distances, x_values, w_values = select_roi_with_distances(img, img_bin)
    #cv2.imshow('Modified Image with Marked Regions', image_orig)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return image_orig, sorted_regions, region_distances, x_values, w_values

#############
#   START   #
#############

'''
data_list=[]
data = Data('captcha_1.jpg', 'стучать стучать') 
data_list.append(data)
data = Data('captcha_2.jpg', 'кто это')  
data_list.append(data)
data = Data('captcha_3.jpg', 'кошатница')  
data_list.append(data)
data = Data('captcha_4.jpg', 'фул давай')  
data_list.append(data)
data = Data('captcha_5.jpg', 'изи катка')  
data_list.append(data)
data = Data('captcha_6.jpg', 'беспозвоночное')  
data_list.append(data)
data = Data('captcha_7.jpg', 'юность щенок')  
data_list.append(data)
data = Data('captcha_8.jpg', 'ягода ёж')  
data_list.append(data)
data = Data('captcha_9.jpg', 'голубой экран')  
data_list.append(data)
data = Data('captcha_10.jpg', 'хороший въезд')  
data_list.append(data)
'''

data_list=[]
with open((sys.argv[1]+'res.csv'), 'r', encoding='utf-8') as csv_file:    
    csv_reader = csv.reader(csv_file)   
    next(csv_reader)    
    for row in csv_reader:
        image, solution = row
        data_instance = Data(image, solution)
        data_list.append(data_instance)

'''
data_counter=0
sorted_regions_all = []
for data in data_list:
    data_counter+=1
    img = load_image('K3/data1/pictures/' + data.image)
    img_gs = image_gray(img)
    img_bin = image_bin(img_gs)
    if(data_counter in (3, 4, 5, 6, 8)):
        img_bin= erode(img_bin)
    image_orig, sorted_regions, region_distances= select_roi_with_distances(img, img_bin)
    sorted_regions_all += sorted_regions
    cv2.imshow('Modified Image with Marked Regions', image_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''

sorted_regions_all = []

###10###
img = load_image('data1/pictures/' + data_list[9].image)
img_gs = image_gray(img)
img_bin = image_bin(img_gs)
#img_bin= erode(img_gs)
image_orig, sorted_regions_10, region_distances, x, w= select_roi_with_distances(img, img_bin)
del sorted_regions_10[5]
del sorted_regions_10[3]
print(f'broj regiona {len(sorted_regions_10)}')
#cv2.imshow('Modified Image with Marked Regions', image_orig)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

###9###
img = load_image('data1/pictures/' + data_list[8].image)
img_gs = image_gray(img)
img_bin = image_bin(img_gs)
#img_bin= erode(img_gs)
image_orig, sorted_regions_9, region_distances, x, w= select_roi_with_distances(img, img_bin)
del sorted_regions_9[9]
del sorted_regions_9[6]
del sorted_regions_9[5]
del sorted_regions_9[1]
print(f'broj regiona {len(sorted_regions_9)}')
#cv2.imshow('Modified Image with Marked Regions', image_orig)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

###8###
img = load_image('data1/pictures/' + data_list[7].image)
img_gs = image_gray(img)
img_bin = image_bin(img_gs)
img_bin= erode(img_gs)
image_orig, sorted_regions_8, region_distances, x, w= select_roi_with_distances(img, img_bin)
del sorted_regions_8[4]
del sorted_regions_8[3]
del sorted_regions_8[2]
del sorted_regions_8[1]
print(f'broj regiona {len(sorted_regions_8)}')
#cv2.imshow('Modified Image with Marked Regions', image_orig)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

###7###
img = load_image('data1/pictures/' + data_list[6].image)
img_gs = image_gray(img)
img_bin = image_bin(img_gs)
#img_bin= erode(img_gs)
image_orig, sorted_regions_7, region_distances, x, w= select_roi_with_distances(img, img_bin)
del sorted_regions_7[10]
del sorted_regions_7[9]
del sorted_regions_7[8]
del sorted_regions_7[7]
del sorted_regions_7[2]
del sorted_regions_7[1]
print(f'broj regiona {len(sorted_regions_7)}')
#cv2.imshow('Modified Image with Marked Regions', image_orig)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

###6###
img = load_image('data1/pictures/' + data_list[5].image)
img_gs = image_gray(img)
img_bin = image_bin(img_gs)
img_bin= erode(img_gs)
image_orig, sorted_regions_6, region_distances, x, w= select_roi_with_distances(img, img_bin)
del sorted_regions_6[13]
del sorted_regions_6[12]
del sorted_regions_6[11]
del sorted_regions_6[9]
del sorted_regions_6[8]
del sorted_regions_6[7]
del sorted_regions_6[6]
del sorted_regions_6[5]
del sorted_regions_6[4]
del sorted_regions_6[2]
del sorted_regions_6[1]
del sorted_regions_6[0]
print(f'broj regiona {len(sorted_regions_6)}')
#cv2.imshow('Modified Image with Marked Regions', image_orig)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

###4###
img = load_image('data1/pictures/' + data_list[3].image)
img_gs = image_gray(img)
img_bin = image_bin(img_gs)
img_bin= erode(img_gs)
image_orig, sorted_regions_4, region_distances, x, w= select_roi_with_distances(img, img_bin)
del sorted_regions_4[6]
del sorted_regions_4[5]
del sorted_regions_4[4]
del sorted_regions_4[3]
del sorted_regions_4[2]
del sorted_regions_4[1]
print(f'broj regiona {len(sorted_regions_4)}')
#cv2.imshow('Modified Image with Marked Regions', image_orig)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

###3###
img = load_image('data1/pictures/' + data_list[2].image)
img_gs = image_gray(img)
img_bin = image_bin(img_gs)
img_bin= erode(img_gs)
image_orig, sorted_regions_3, region_distances, x, w= select_roi_with_distances(img, img_bin)
del sorted_regions_3[8]
del sorted_regions_3[6]
del sorted_regions_3[5]
del sorted_regions_3[4]
del sorted_regions_3[3]
del sorted_regions_3[2]
del sorted_regions_3[1]
del sorted_regions_3[0]
print(f'broj regiona {len(sorted_regions_3)}')
#cv2.imshow('Modified Image with Marked Regions', image_orig)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

sorted_regions_all+=sorted_regions_10 + sorted_regions_9 + sorted_regions_8 + sorted_regions_7 + sorted_regions_6 + sorted_regions_4 + sorted_regions_3
#'''
alphabet = ['х', 'о', 'р', 'ш', 'и', 'в', 'ъ', 'е', 'з', 'д', 'г', 'л', 'у', 'б', 'э', 'к', 'а', 'н', 'я', 'ё', 'ж', 'ю', 'с', 'т', 'ь', 'щ', 'п', 'ч', 'ф', 'й', 'ц']
inputs = prepare_for_ann(sorted_regions_all)
outputs = convert_output(alphabet)
ann = create_ann(output_size=31)
ann = train_ann(ann, inputs, outputs, epochs=800)


data_counter=0
prediction_list=[]
for data in data_list:
    data_counter+=1
    img = load_image('data1/pictures/' + data.image)
    img_gs = image_gray(img)
    img_bin = image_bin(img_gs)
    if(data_counter == 1):
       img_bin= erode_two(img_gs)
    image_orig, sorted_regions, region_distances, x, w= select_roi_with_distances(img, img_bin)
    if(data_counter == 8):
       img_bin= erode_two(img_gs)
       image_orig, sorted_regions, region_distances, x, w = erode_test(img_bin, 5) 
    region_distances = np.array(region_distances).reshape(len(region_distances), 1)
    k_means = KMeans(n_clusters=2)
    k_means.fit(region_distances)
    inputs = prepare_for_ann(sorted_regions)
    results = ann.predict(np.array(inputs, np.float32))
    print("Broj prepoznatih regiona: ", len(sorted_regions))
    if(data_counter not in (3, 6)):
        print(display_result_with_spaces(results, alphabet, k_means))
        prediction_list.append(display_result_with_spaces(results, alphabet, k_means))
    else:
        print(display_result(results, alphabet))
        prediction_list.append(display_result(results, alphabet))
    #cv2.imshow('Modified Image with Marked Regions', image_orig)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

#print(prediction_list)

sum_distance = 0
counter=0
for data in data_list:
    sum_distance += hamming_distance(data.solution, prediction_list[counter])
    print(f"{data.image}-{data.solution}-{prediction_list[counter]}")
    counter+=1
print(f"DISTANCE SUM: {sum_distance}")