import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import json
import math
import face_alignment
import xml.etree.ElementTree as ET
import numpy as np
import json
from matplotlib import pyplot as plt
from utils.landmark_util import extract_landmarks_media_pipe
from scipy import signal
from scipy.stats import multivariate_normal
from scipy.spatial import distance
from utils.motion_extraction_util import *
import cv2
from utils.canonical_face import ObjLoader
from qpsolvers import solve_qp

def extract_normalized_landmark(video_file_name, video_file_folder_location, lmPath=None):

    # the output of the landmark will be located inside a folder with the same name as
    # the video file, named "normalized_mediapipe_landmark.npy"

    if lmPath is None:
        extract_landmarks_media_pipe(video_file_name,
                                     video_file_folder_location,
                                     save_annotated_video=False)
        lmPath = video_file_folder_location + video_file_name[:-4] + "/" + "raw_mediapipe_landmark.npy"
    videoPath = video_file_folder_location + video_file_name
    with open("mediaPipeMapping.json", "r") as f:
        maping = json.load(f)

    staticLandmarkIndices = maping["nose"]["dorsum"] + maping["nose"]["tipLower"] + maping["additional_anchors"]
    keypointIndicies = maping["nose"]["dorsum"] + maping["nose"]["tipLower"] + maping["additional_anchors"] + \
                       maping["brow"]["rightLower"] + maping["brow"]["rightUpper"] + maping["brow"]["leftUpper"] + \
                       maping["brow"]["leftLower"] + maping["eye"]["right"] + maping["eye"]["left"] + maping["lips"][
                           "inner"] + maping["lips"]["outer"]
    data = np.load(lmPath)
    cap = cv2.VideoCapture(videoPath)
    t = 0
    flow_data = []
    video_shape = []
    while cap.isOpened() and t < data.shape[0]:
        # obtain the image frame
        ret, frame_t1 = cap.read()
        if frame_t1 is None:
            break
        # obtain the landmark frame:
        lm_t1 = data[t - 1, keypointIndicies].astype(np.float32)

        # un-normalize the position by multiplying by the frame size
        lm_t1[:, 0] = lm_t1[:, 0] * frame_t1.shape[1]
        lm_t1[:, 1] = lm_t1[:, 1] * frame_t1.shape[0]
        ##### calculate optical flow #####
        # convert to gray scale
        frame_t1_gray = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2GRAY)
        lm_t1 = np.expand_dims(lm_t1, axis=1)[:, :, 0:2]
        if t == 0:
            video_shape = frame_t1_gray.shape
            frame_t0_gray = frame_t1_gray.copy()

            lm_t1 = data[0, keypointIndicies].astype(np.float32)
            # un-normalize the position by multiplying by the frame size
            lm_t1[:, 0] = lm_t1[:, 0] * frame_t1.shape[1]
            lm_t1[:, 1] = lm_t1[:, 1] * frame_t1.shape[0]
            ##### calculate optical flow #####
            # convert to gray scale
            frame_t1_gray = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2GRAY)
            lm_t1 = np.expand_dims(lm_t1, axis=1)[:, :, 0:2]
            p_t0 = lm_t1

            flow_data.append(p_t0[:, 0, :])
            t = t + 1
            continue
        p_t1, st, err = cv2.calcOpticalFlowPyrLK(
            frame_t0_gray, frame_t1_gray, p_t0, None, **lk_params
        )
        # Select good points
        good_new = p_t1[st == 1]
        good_old = p_t0[st == 1]
        flow_data.append(np.where(st == 1, p_t1[:, 0, :], p_t0[:, 0, :]))
        t = t + 1
        frame_t0_gray = frame_t1_gray.copy()
        p_t0 = lm_t1

    cap.release()
    # re-scale the optical flow data
    flow_data = np.array(flow_data)
    flow_data[:, :, 0] = flow_data[:, :, 0] / video_shape[1]
    flow_data[:, :, 1] = flow_data[:, :, 1] / video_shape[0]
    flow_data_3d = data[:, keypointIndicies].copy()
    flow_data_3d[:, :, 0:2] = flow_data
    alpha = 1 / 2.2
    beta = 0.2 / 2.2
    gamma = 2 / 2.2
    lm_data_3d = data[:, keypointIndicies]
    flow_data_3d = flow_data_3d
    face = ObjLoader("./data/canonical_face_model.obj")
    lm_data_to_canonical = iterativeNormalization(data, face.vertices, staticLandmarkIndices, staticLandmarkIndices)[:,
                           keypointIndicies]
    flow_data_3d_to_canonical = iterativeNormalization(flow_data_3d, face.vertices[keypointIndicies],
                                                       np.arange(0, len(staticLandmarkIndices)),
                                                       np.arange(0, len(staticLandmarkIndices)))

    lm_data_to_canonical = rotateToNeutral(face.vertices, data, staticLandmarkIndices)[:, keypointIndicies]
    N_landmakrs, N_dims = lm_data_to_canonical[0].shape
    qp_sols = [lm_data_to_canonical[0].reshape(N_landmakrs * N_dims)]
    for i in range(1, lm_data_to_canonical.shape[0]):
        L_lm_t = lm_data_to_canonical[i].reshape(N_landmakrs * N_dims)
        L_fl_t = flow_data_3d_to_canonical[i].reshape(N_landmakrs * N_dims)
        L_lm_prev = qp_sols[-1] / 2
        P = (alpha + beta + gamma) * np.eye(N_landmakrs * N_dims)
        q = -2 * (alpha * L_lm_t + beta * L_fl_t + gamma * L_lm_prev)
        L = solve_qp(P, q, initvals=L_lm_t)
        qp_sols.append(L)
    L_stabilized = np.array(qp_sols)
    L_stabilized = L_stabilized.reshape((L_stabilized.shape[0], N_landmakrs, N_dims))
    return L_stabilized
if __name__ == "__main__":
    extract_normalized_landmark("rolling_in_the_deep_1.mp4", "F:/MASC/Motion_paint/example_videos/", "F:/MASC/Motion_paint/example_videos/rolling_in_the_deep_1/raw_mediapipe_landmark.npy")
