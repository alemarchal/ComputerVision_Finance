#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:26:56 2020

@author: alexismarchal
"""



# TO RUN IN THE TERMINAL:
#    python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method hog



# Encoding the faces using OpenCV and deep learning


# <==> Quantify the faces in our training set

# Use the pre-trained network and then use it to construct 128-d embeddings
# for each of the faces in our dataset.



# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os



# construct the argument parser and parse the arguments (for the terminal command line)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# --encodings : Our face encodings are written to the file that this argument points to.

# --detection-method : Before we can encode faces in images we first need to detect them. 
#   Or two face detection methods include either hog  or cnn . Those two flags are the only ones that will work for --detection-method

# The CNN method is more accurate but slower. HOG is faster but less accurate.




# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"])) #build a list of all imagePaths contained therein


# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []



# loop over the image paths
for (i, imagePath) in enumerate(imagePaths): #We’re looping over the paths to each of the images
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2] #To get the correct name, our subdirectory should be named appropriately
    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #OpenCV orders color channels in BGR, but the dlib  actually expects RGB. The face_recognition  module uses dlib , so before we proceed, let’s swap color spaces


    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb,
        model=args["detection_method"])
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    # Here, we’re going to turn the bounding boxes  of the person’s face into a list of 128 numbers. 
    # This is known as encoding the face into a vector and
    # the face_recognition.face_encodings method handles it for us.
    
    
    
    
    
    
    
    # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and
        # encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

# For each iteration of the loop, we’re going to detect a face 
# (or possibly multiple faces and assume that it is the same person in multiple 
# locations of the image — this assumption may or may not hold true in your own 
# images so be careful here).





# Now we export the results so that we will use 
# the encodings in another script which handles the recognition




# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames} #constructs a dictionary with two keys: "encodings"  and "names"
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()









