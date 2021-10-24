#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 22:49:39 2020

@author: alexismarchal
"""

# This code takes as input the time-series of EAR of a video and processes it.


import glob
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle # To save variables as external files
from datetime import datetime
import os
path_source_files = os.getcwd()+''+'/saved outputs/saved outputs_ear v1'


# Initiate empty lists
date_file=[]
length_speech_chairman=[] #Length of the speech of the chairman ONLY (talks by reporters is not included)
sum_ear_below_threshold=[]
length_ear_below_threshold=[]

for file_name in glob.glob(os.path.join(path_source_files, '*')):

    # Extract the date of the associated video
    sep = '-' #separator
    temp_date=file_name.split(sep, -1)[-1] #split at all '-' and keep the last part
    date_file.append(temp_date)


    with open(file_name, 'rb') as f:
        ear_with_nan = pickle.load(f)
    
    
    #plt.plot(ear.index,ear['ear'])
    #plt.xlabel('time (seconds)')
    #plt.ylabel('Eye aspect ratio')
    
    
    ear = ear_with_nan.dropna() # Remove the NaN observations
    ear = ear.reset_index(drop=True)
    

    # What to do with the observations when I can detect the Chairman face but can't detect the eyes?
    ear = ear.drop(ear[ear['ear'] == -1].index) # Remove observations when EAR = -1
    ear = ear.reset_index(drop=True)


#    ear = ear.replace(to_replace = -1, value = -0.0) #Replace the -1 observations with another value




    # I summazire the entire length of the Press Conference in one real value
    ear_below_threshold = ear.drop(ear[ear['ear'] > 0.276].index)
    length_ear_below_threshold.append( len(ear_below_threshold['ear']) )
    sum_ear_below_threshold.append(ear_below_threshold['ear'].sum())



    # The dataframe ear['ear'] contains an entry every time I could see the chairman talking.
    # Since I have removed the NaNs, the length of this vector corresponds approximately
    # to how long the chairman spoke during the press conference.
    length_speech_chairman.append(len(ear['ear']))


# Convert the dates to a standard format
date_time_obj = [datetime.strptime(date, '%d_%B_%Y').date() for date in date_file]

# Create DataFrame
df_aggregate_ear =  pd.DataFrame({'Date':date_time_obj})


# Store all the variables
df_aggregate_ear['length_ear_below_threshold'] = length_ear_below_threshold
df_aggregate_ear['sum_ear_below_threshold'] = sum_ear_below_threshold
df_aggregate_ear['length_speech_chairman'] = length_speech_chairman



# Sort everything by date
df_aggregate_ear = df_aggregate_ear.sort_values(by='Date')
df_aggregate_ear = df_aggregate_ear.reset_index(drop=True)


with open('df_aggregate_ear', 'wb') as f:
    pickle.dump(df_aggregate_ear, f)





