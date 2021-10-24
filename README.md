# Computer Vision in Finance

This code analyzes the nonverbal communication of FED officials using FOMC meetings videos.
This repository contains the main parts of the code necessary to replicate the findings of my [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3747172).

Parts of this code are modification of tutorials found on the [PyImageSearch blog](https://www.pyimagesearch.com) (see references).

This code will give you a time series of Eye Aspect Ratio (EAR) for each video. The scripts explaining stock returns and volatility are not included but can be accessed on demand (for instance send me a message on LinkedIn).

Give it a shot by running "detect_eyes.py" !

## Important notes

The encodings file (pickle format) contains the encoding of the faces of the FOMC Chairs since the videos started to be published. If you wish to use this code on videos featuring different people, you will have to encode their faces by creating a dataset of images and running the corresponding script (this folder of images needs a particular structure, see references for more info).
I do not upload the images used for face encoding in this repo given their size.

You will also need the "shape_predictor_68_face_landmarks.dat" file available for instance on the PyImageSearch blog and that I do not upload (~100MB).



## References

The two tutorials that were used are:

Adrian Rosebrock, Eye blink detection with OpenCV, Python, and dlib, PyImageSearch, https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/, accessed on 28 August 2020


Adrian Rosebrock, Face recognition with OpenCV, Python, and deep learning, PyImageSearch, https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/, accessed on 28 October 2020


