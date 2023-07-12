import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_copying_tests import C_T
from sklearn.cluster import KMeans
from keras.datasets import cifar10
import os
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import cv2
#matplotlib.use('TkAgg')

def add_noise(image, i):
    noisy_image = image
    shape = noisy_image.shape
    noise = np.random.normal(0, 0.05, shape)
    noisy_image = cv2.add(image, noise)
    noisy_image = np.copy(noisy_image)
    return noisy_image

def test(P, Q, T):
    list_of_noisy_samples = [Q]
    scores = []
    for i in range(10):
        noisy_images = []
        print("Preparing noisy sample", i+1)
        for image in list_of_noisy_samples[i]:
            noisy_image = add_noise(image, i)
            noisy_image = noisy_image.flatten()
            #print("ni shape", noisy_image.shape)
            noisy_images.append(noisy_image)
        list_of_noisy_samples.append(noisy_images)
    n_clusters = 10 # of cells
    KM_clf = KMeans(n_clusters).fit(T)
    Pn_cells = KM_clf.predict(P)
    T_cells = KM_clf.predict(T)
    for sample in list_of_noisy_samples:
        print('Getting instance space partition...')
        #flattened_sample = np.array([image.flatten() for image in sample])
        sample = np.array(sample)
        Qm_cells = KM_clf.predict(sample)#duplicate cell labels are allowed becuase they simply divide up the data
        ct, zu = C_T(P, Pn_cells, sample, Qm_cells, T, T_cells, tau = 20 / len(sample))
        scores.append(ct)
    x_axis = [0,1,2,3,4,5,6,7,8,9,10]
    plt.plot(x_axis, scores, marker='o')
    plt.xlabel("Iterations")
    plt.ylabel("CT scores")
    plt.title("CT vs Noisy Test Sample")
    plt.savefig('ct_v_noise.png')
    return scores

def plot_for_multiple_paths(paths, testX, trainX):
    path_to_gen = []
    truncs = []
    for path in paths:
        path_to_gen.append(path[0])
        truncs.append(path[1])
    scores = []
        # Iterate through each file in the directory
    for path in path_to_gen:
        gen = []
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):
                filepath = os.path.join(path, filename)        
            # Load the image and convert it to a numpy array
                image = Image.open(filepath)
                image_array = np.array(image)
            
            # Add the image array to X
                gen.append(image_array)
            
            # Extract the label from the filename (assuming the filename format: label_image.jpg)
        #label = filename.split("_")[0]
        #y.append(label)
    
    # Convert the lists to numpy arrays
        gen = np.array(gen)
        gen = gen.reshape(gen.shape[0], -1)
        gen = scaler.transform(gen)

        Pn = testX  # Use testX as the test sample
        Qm = gen #needs to be generated samples
        T = trainX
    #Qm = trainX #incorrect, only for testing purposes

    #get instance space partition
        print('Getting instance space partition...')
        n_clusters = 10 # of cells
        KM_clf = KMeans(n_clusters).fit(trainX)
        Pn_cells = KM_clf.predict(Pn)
        T_cells = KM_clf.predict(T)
        Qm_cells = KM_clf.predict(Qm)#duplicate cell labels are allowed becuase they simply divide up the data
        print("# of labels for P: ", len(Pn_cells), "; Q: ", len(Qm_cells), "; T: ", len(T_cells))
        print("size of data for P: ", len(Pn), "; Q: ", len(Qm), "; T: ", len(T))
        '''print("P labels: ", Pn_cells)
        print("Q labels: ", Qm_cells)
        print("T labels: ", T_cells)
        '''
        # Generate unique cell labels for each partition
        train_cells = np.arange(trainX.shape[0])  # Cell labels for trainX: 0 to (trainX.shape[0] - 1)
        val_cells = np.arange(valX.shape[0]) + trainX.shape[0]  # Cell labels for valX: trainX.shape[0] to (trainX.shape[0] + valX.shape[0] - 1)
        test_cells = np.arange(testX.shape[0])  # Cell labels for testX: 0 to (testX.shape[0] - 1)'''

    #Pn_cells = test_cells  # Cell labels for testX

        ct, zu = C_T(Pn, Pn_cells, Qm, Qm_cells, T, T_cells, tau = 20 / len(Qm)) #tau is the threshold fraction of samples from Q that must exist in a cell for it to be evaluated by the metric
        scores.append(ct)
    plt.plot(truncs, scores, marker='o')
    plt.xlabel("Truncation Thresholds")
    plt.ylabel("CT scores")
    plt.title("CT vs Truncation")
    plt.savefig('ct_v_trnc.png')
    return scores

#get samples
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.reshape(trainX.shape[0], -1) #transforms each image into a vector
testX = testX.reshape(testX.shape[0], -1)
scaler = MinMaxScaler()
trainX = scaler.fit_transform(trainX)#estimates the scaling parameters and transforms
testX = scaler.transform(testX)#transforms based on previous scaling parameters
val_data_size = 10000  # adjust the size as needed
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=val_data_size, random_state=42)

path1 = ('/home/nburger/FastDPM_pytorch/generated/var_b/var_b_5_128', 5)#paths to different generated samples
path2 = ('/home/nburger/FastDPM_pytorch/generated/var_b/var_b_4_128', 4)
path3 = ('/home/nburger/FastDPM_pytorch/generated/var_b/var_b_3_128', 3)
path4 = ('/home/nburger/FastDPM_pytorch/generated/var_b/var_b_2_128', 2)
path5 = ('/home/nburger/FastDPM_pytorch/generated/var_b/var_b_1_128', 1)
path6 = ('/home/nburger/FastDPM_pytorch/generated/var_b/var_b_.5_128', .5)
path7 = ('/home/nburger/FastDPM_pytorch/generated/var_b/var_b_.1_128', .1)
path8 = ('/home/nburger/FastDPM_pytorch/generated/var_b/var_b_.05_128', .05)
path9 = ('/home/nburger/FastDPM_pytorch/generated/var_b/var_b_.01_128', .01)

paths = [path1, path2, path3, path4, path5, path6, path7, path8, path9]
scores = plot_for_multiple_paths(paths, testX, trainX)
#scores = test(testX, trainX, trainX)
print("CT scores: ", scores)


'''
print("Zu: ", zu)
print("C_T: ", ct)
if(ct < -5):
    print("Some degree of overfitting is highly likely.")
elif(ct > 5):
    print("Some degree of under-fitting is highly likely, make sure you're using the correct data samples.")
elif(ct > 0):
    print("A small degree of underfitting may be present.")
elif(ct < 0):
    print("A small degree of overfitting may be present.")
'''