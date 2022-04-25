from utils.motion_extraction_util import *
import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import math
import face_alignment
import numpy as np
import json
from utils.canonical_face import *
from matplotlib import pyplot as plt
from utils.landmark_util import extract_landmarks_media_pipe
from scipy import signal
from scipy.stats import multivariate_normal
from scipy.spatial import distance
if __name__ == "__main__":
    lmPath = "F:/MASC/Motion_paint/example_videos/rolling_in_the_deep_1/raw_mediapipe_landmark.npy"
    videoPath = "F:/MASC/Motion_paint/example_videos/rolling_in_the_deep_1.mp4"

    # lmPath = "/Volumes/EVAN_DISK/MASC/Motion_paint/example_videos/Child_in_time_2/raw_mediapipe_landmark.npy"
    # lmPath = "E:/Facial Feature Motion Clip/rollingInTheDeep/raw_mediapipe_landmark.npy"
    # lmPath = "/Volumes/KINGSTON/csc2521/raw_mediapipe_landmark.npy"
    # outputPath = "/Volumes/EVAN_DISK/emp/child_in_time_motion.json"
    outputPath = "C:/Users/evansamaa/Desktop/Motion_Paint/facial_landmarks_py/data/motion.json"
    rotation_output_path = "C:/Users/evansamaa/Desktop/Motion_Paint/facial_landmarks_py/data/Rot_motion.json"
    # info from video
    t_segment = [0, 20]
    t_ref = 0
    # info of target animation
    start = 0
    fps = 30
    min_cutoff = 0.0001
    beta = 1.5
    with open("mediaPipeMapping.json", "r") as f:
        maping = json.load(f)
    staticLandmarkIndices = maping["nose"]["dorsum"]+maping["nose"]["tipLower"]+maping["additional_anchors"]
    keypointIndicies = maping["nose"]["dorsum"]+maping["nose"]["tipLower"]+maping["additional_anchors"]+maping["brow"]["rightLower"]+maping["brow"]["rightUpper"]+maping["brow"]["leftUpper"]+maping["brow"]["leftLower"]+maping["eye"]["right"]+maping["eye"]["left"]+maping["lips"]["inner"]+maping["lips"]["outer"]
    data = np.load(lmPath)

    face = ObjLoader("./data/canonical_face_model.obj")
    lm_data_to_canonical = iterativeNormalization(data[t_segment[0]:t_segment[1]], face.vertices, staticLandmarkIndices, staticLandmarkIndices)
    i = 19
    plt.scatter(lm_data_to_canonical[i, keypointIndicies, 0],
                lm_data_to_canonical[i, keypointIndicies, 1])
    plt.scatter(face.vertices[keypointIndicies][:, 0], face.vertices[keypointIndicies][:, 1])
    plt.show()