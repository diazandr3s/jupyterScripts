
import glob
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg') # this is for saving the images using a non-interactive backend Agg
import matplotlib.pyplot as plt
import math
import subprocess
import time


# Reading csv files

# 1: image file name
# 2-5: detected optic disc bounding box coordinates, i.e., [x1, y1, x2, y2]
# 6-9: detected optic cup bounding box coordinates, i.e., [x1, y1, x2, y2]
# 10: confidence score for the detected optic disc bounding box
# 11: confidence score for the detected optic cup bounding box
# 12: image quality assessment output score
# 13: image quality assessment results, where 1 indicates good quality and 0 otherwise.
#     The threshold is set to be 0.5 here.

db = 'KAGGLE'
infoTrain = pd.read_csv('./info2crop/' + db + '/train.csv', sep=',', header=None)
infoTest = pd.read_csv('./info2crop/' + db + '/test.csv', sep=',', header=None)

infoAux = [infoTrain, infoTest]
info = pd.concat(infoAux)

names = info.iloc[:,0].tolist()

# x1, y1 is the center. NOT SURE ABOUT THIS
# make the execution secure. There are some values equal to zero. Maybe checking the quality assessment value
x1 = info.iloc[:,1].tolist()
y1 = info.iloc[:,2].tolist()
x2 = info.iloc[:,3].tolist()
y2 = info.iloc[:,4].tolist()
scoreQ = info.iloc[:,11].tolist()
scoreOD = info.iloc[:,9].tolist()


for i in range(len(y2)):

    start_time = time.time()
            
    im = cv2.imread('../DCGAN_UNIT_baseline/S3_bucket/images2crop/' + db + '/' + names[i])
    
    rad = math.hypot(x2[i] - x1[i], y2[i] - y1[i]) / 3  # Linear distance

    imCrop = im[int(y1[i] - rad):int(y2[i] + rad), int(x1[i] - rad):int(x2[i] + rad), :]

    # Saving cropped images
    cv2.imwrite("../DCGAN_UNIT_baseline/S3_bucket/imagesCropped/" + db + "/" + names[i], imCrop)
    
    print('Processing Image number ' + str(i) + ' --- Quality score ' + str(scoreQ[i]) + '  --- OD score ' + str(scoreOD[i]) + " --- %s seconds to crop and save image ---" % (time.time() - start_time) )
        
    
    