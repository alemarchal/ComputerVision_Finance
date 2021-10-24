#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alexismarchal
"""

import os
import glob


# Libraries used to parralelize the loop (on different CPUs)
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()



# This code just loops over all the videos from the FOMC press conferences



path_videos_FOMC='/Users/alexismarchal/Research Projects/Q&A only videos FOMC Press Conferences'


file_names=[]

#build a list of all the videos in my directory
for x in glob.glob(os.path.join(path_videos_FOMC, '*.mp4')):
    file_names.append(x)


# i=1
#     print("[INFO] processing video {}/{}".format(i ,len(file_names)))
#     i=i+1


def processInput(file_name):
    os.system("python detect_eyes.py --shape-predictor shape_predictor_68_face_landmarks.dat --video '{}' --display_video True --print_ear False".format(file_name))


Parallel(n_jobs=num_cores)(delayed(processInput)(n) for n in file_names)





