#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:17:47 2020

@author: alexismarchal
"""

# This function takes as input a single image (a frame from a video for instance)
# and outputs the identity of the person on the image.



# Necessary packages
import face_recognition
import cv2



def name_in_frame(image,encodings_from_file,detection_method,chosen_tolerance):

     # Recognizing faces in images
    """
    Important Performance Note: The CNN face recognizer should only be used in 
    real-time if you are working with a GPU (you can use it with a CPU, but expect 
    less than 0.5 FPS which makes for a choppy video). 
    Alternatively (you are using a CPU), you should use the HoG method
    (or even OpenCV Haar cascades covered in a future blog post) and expect adequate speeds. 
    """
    



    
    # let’s load the pre-computed encodings + face names and then construct the 128-d face encoding for the input image
    
    # load the known faces and embeddings

    data = encodings_from_file
    # convert the input image from BGR to RGB

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    
    boxes = face_recognition.face_locations(rgb,model=detection_method)
    encodings = face_recognition.face_encodings(rgb, boxes)
    # initialize the list of names for each face detected
    names = []
    
    
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],encoding,tolerance=chosen_tolerance)
        name = "Unknown"
    
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)
    
        # update the list of names
        names.append(name)
    
    
    
    return boxes,names


