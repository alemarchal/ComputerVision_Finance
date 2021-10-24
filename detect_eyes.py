# USAGE

# cd '/Users/alexismarchal/Research Projects/Eye detector'

# python detect_eyes.py --shape-predictor shape_predictor_68_face_landmarks.dat --video test.MOV --display_video True --print_ear True


"""
This file will only detect eyes if the person is a Chair of the FOMC.
If another person is in the video it will output NaN for the EAR. (can be modified easily)
"""


# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


#from beepy import beep # For the sounds
import matplotlib.pyplot as plt
import pickle # To save variables as external files
from moviepy.editor import VideoFileClip # To get the video length
import pandas as pd
import os







#To recognize the identity of the person in the video frame
from function_name_frame import name_in_frame



def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear




"""
To prevent the code in the module from being executed when imported, 
but only when run directly, you can guard it with this:

"""
if __name__ == "__main__":
    # The code below won't be run when imported from another file.
    # If I don't do this, when I import a function from this file,
    # Python will run the whole file (which is not what I want).

    
    
    
    
    # This file contains the encodings from the faces we want to detect.
    # This file is produced by executing the code 'encode_faces.py'
    encodings_from_file = pickle.loads(open('encodings.pickle', "rb").read())
    
    
    detection_method = 'hog'
    
    
    FOMC_chairman_names = ['Ben_Bernanke','Janet_Yellen','Jerome_Powell']
    
    
    
    #The smaller the tolerance, the more strict the facial recognition system will be
    tolerance_faces = 0.5   # 0.6 is standard
    
    
    
    # When the algo cannot detect the eyes, I set the ear to this value
    value_ear_when_cannot_detect_eyes = -1
    
    
    
    # Dimension of the frames in the video. The larger, the more pixels and so the slower is the code.
    dimension_frame = 500   # 500 is a value that works pretty well
                             # I can set it to None if I don't want to resize
    
    
    
    
    
    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,help="path to facial landmark predictor")
    ap.add_argument("-v", "--video", type=str, default="",help="path to input video file")
    ap.add_argument("-y","--display_video",type=str, required=True, help="True or False to display the video")
    ap.add_argument("-d","--print_ear", required=False, type=str, help="True or False for printing the EAR in real time in the console")
    args = vars(ap.parse_args())
     
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold
    EYE_AR_THRESH = 0.225
    EYE_AR_CONSEC_FRAMES = 3
    
    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0
    
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
    
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = FileVideoStream(args["video"]).start()
    fileStream = True
    # vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start()
    # fileStream = False
    time.sleep(1.0)
    
    ear_time_series = []
    
    iter_counter=0
    
    ear = np.nan #initial value for ear in case it's not possible to compute it at 
    # the beginning of the video (because I can't detect the eyes for the first
    # frames for instance).
    
    
    live_plots_folder = os.getcwd()+''+'/Live plots/'
    with open(live_plots_folder+''+'output_EAR_live_plot.txt', 'w') as f: #File initialization
        print('',file=f)
    t=0 # Initialize the variable that will capture the instant "t" when live plotting
    
    
    
    # loop over frames from the video stream
    while True:
    
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if fileStream and not vs.more():
            break
    
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        frame = vs.read()
    
        # With FileVideoStream, When the end of the video file has been reached, 
        # the frame variable contains None. In this case, the frame should
        # not be added to the queue. The (iter_counter) makes sure that the None
        # problem doesn't arise at the beginning of the video (so that the problem 
        # is not with opening the video).
        if frame is None and iter_counter > 1:
            continue
    
        frame = imutils.resize(frame, width=dimension_frame)
    
        # extract the identity of the person(s) on the image as well as
        # the dimension of the box(es) surrounding the face(s)
        boxes,names = name_in_frame(frame,encodings_from_file,detection_method,tolerance_faces)
        nbr_faces = len(names) # Number of different faces detected in one frame
    
        if nbr_faces==1 and names[0] in FOMC_chairman_names:
        # I only compute the EAR if there is a single face detected which happens
        # to be one of the fed chairman. Otherwise I skip this iteration.
    
    
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
            # detect faces in the grayscale frame
            rects = detector(gray, 0)
            #rects will be an empty list if it cannot detect a face (and so it if can't detect the eyes)
            #So when len(rects)=0 when there is no face detected (and so no eyes detected and so it can't compute the EAR)
    
            if len(rects)==0:
                ear=value_ear_when_cannot_detect_eyes
            # This is important because if I don't do that, when the eyes cannot
            # be detected, the next loop won't be executed and the EAR will be set
            # to the previous value (last time the eyes were detected).
            # This is the value of EAR when the Central Bank chairman face is detected
            # but the eyes cannot be detected (for instance when he looks down).
    
    
            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
    
                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
    
                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0
    
                if args["display_video"] == 'True':
                    # compute the convex hull for the left and right eye, then
                    # visualize each of the eyes
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    
    
                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
    
                # otherwise, the eye aspect ratio is not below the blink
                # threshold
                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
    
        #               beep(sound=1)
        
                    # reset the eye frame counter
                    COUNTER = 0
    
                # draw the total number of blinks on the frame along with
                # the computed eye aspect ratio for the frame
                if args["display_video"] == 'True':
            #       cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
                    #loop over the recognized faces
                    for ((top, right, bottom, left), name) in zip(boxes, names):
                        # draw the predicted face name on the image
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        y = top - 15 if top - 15 > 15 else top + 15
                        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
    
    
        else:# Do nothing in case I detect more or less than 1 person or it's not a fed chairman on the image.
            ear=np.nan 
            # This is the EAR when the Central Bank chairman face is not detected at all (he is not on the frame).
    
    
        # show the frame
        if args["display_video"] == 'True':
            cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
     
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    
        ear_time_series.append(ear)
        iter_counter=iter_counter+1
    
        if args["print_ear"] == 'True':
            print('EAR =',ear) # printing the variable in the Terminal

            t=t+1
            with open(live_plots_folder+''+'output_EAR_live_plot.txt', 'a') as f: # printing in an external file
                print(t,",",ear,file=f) # print(x,y) where x is the time instant t and y is the value I plot
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    
    
    sep_1 = '/' #separator 1
    temp_1 = args["video"].split(sep_1, -1)[-1]
    
    
    # Get rid of the extension in the name of the video
    sep = '.' #separator
    video_name_no_extension = temp_1.split(sep, 1)[0]
    
    
    
    # Create a subfolder to store the outputs (if it doesn't already exist)
    path_outputs_ear = os.getcwd()+''+'/outputs_ear'
    if not os.path.exists(path_outputs_ear):
        os.makedirs(path_outputs_ear)
    
    
    # Create a subfolder to store the outputs (if it doesn't already exist)
    path_outputs_plots = os.getcwd()+''+'/outputs_plots'
    if not os.path.exists(path_outputs_plots):
        os.makedirs(path_outputs_plots)
    
    
    
    # getting the length of the video (in seconds)
    clip = VideoFileClip(args["video"])
    time_axis = np.linspace(0, clip.duration, len(ear_time_series)) #time axis
    
    
    plt.plot(time_axis,ear_time_series)
    plt.xlabel('time (seconds)')
    plt.ylabel('Eye aspect ratio')
    plt.savefig(path_outputs_plots+''+'/ear_{0}.png'.format(video_name_no_extension))
    
    
    # Eye Aspect Ratio indexed by the time in the video
    ear_final = pd.DataFrame(index=time_axis)
    ear_final['ear'] = ear_time_series
    ear_final['clip_length'] = clip.duration
    
    with open(path_outputs_ear+''+'/ear_{0}'.format(video_name_no_extension), 'wb') as f:
        pickle.dump(ear_final, f)
    
    
    
    
    






