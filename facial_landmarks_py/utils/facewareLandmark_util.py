import xml.etree.ElementTree as ET
import numpy as np
from matplotlib import pyplot as plt
from utils.landmark_util import extract_landmarks_media_pipe

class faceWareLandmarkObject():

    fps = 0
    start = 0
    end = 0
    total_frame = 0
    num_landmarks = 52
    data = None
    data_dict = {"mouth":list(range(25, 51)), "eye":list(range(13, 25)), "nose":[0, 1, 2], "brow":list(range(3, 13))}
    def __init__(self, fileName):
        # [metadata/frames][frame_i][0][mouth/eyes/brows][0][landmark_i]
        mytree = ET.parse(fileName)
        myroot = mytree.getroot()
        self.start = myroot[0][0].attrib["rangeStart"]
        self.end = myroot[0][0].attrib["rangeEnd"]
        self.fps = int(np.ceil(float(myroot[0][0].attrib["frameRate"])))
        self.total_frame = faceWareLandmarkObject.stringToFrame(self.end, self.fps)
        i = 3
        self.data = []
        for t in range(0, self.total_frame):
            frameT = np.zeros((52, 3))
            for i in range(0, 13):# nose and brow
                # [metadata/frames][frame_i][0][mouth/eyes/brows][0][landmark_i][textCoord2D/position3D]
                u_i = float(myroot[1][t][0][2][0][i][0].attrib["u"])
                v_i = -float(myroot[1][t][0][2][0][i][0].attrib["v"])
                frameT[i, 0] = u_i
                frameT[i, 1] = v_i
            for i in range(3, 15):  # nose and brow
                # [metadata/frames][frame_i][0][mouth/eyes/brows][0][landmark_i][textCoord2D/position3D]
                u_i = float(myroot[1][t][0][1][0][i][0].attrib["u"])
                v_i = -float(myroot[1][t][0][1][0][i][0].attrib["v"])
                frameT[i+10, 0] = u_i
                frameT[i+10, 1] = v_i
            for i in range(0, 27):  # nose and brow
                # [metadata/frames][frame_i][0][mouth/eyes/brows][0][landmark_i][textCoord2D/position3D]
                u_i = float(myroot[1][t][0][0][0][i][0].attrib["u"])
                v_i = -float(myroot[1][t][0][0][0][i][0].attrib["v"])
                frameT[25+i, 0] = u_i
                frameT[25+i, 1] = v_i
            self.data.append(np.expand_dims(frameT, 0))
        self.data = np.concatenate(self.data, axis=0)
        print(self.data.shape)
    @staticmethod
    def stringToFrame(strFrameVal, fps):
        minutes = strFrameVal.split(":")[1]
        sec = strFrameVal.split(":")[2]
        frames = strFrameVal.split(":")[3]
        out = (int(minutes) * 60 + int(sec)) * fps + int(frames) + 1
        return out
    def getBrows(self):
        leftBrow = self.data[:, 8:13, :]
        rightBrow = self.data[:, 3:8, :]
        return [leftBrow, rightBrow]
    def getNose(self):
        nose = self.data[:, 0:3, :]
        return nose
    def getEyes(self):
        rightEye = self.data[:,13:17]
        leftEye = self.data[:, 17:21]
        return [rightEye, leftEye]
    def getMouth(self):
        innerLip = self.data[:, 39:52]
        outerLip = self.data[:, 25:39]
        return [innerLip, outerLip]
if __name__ == "__main__":
    # with open("mediaPipeMapping.json", "r") as f:
    #     maping = json.load(f)
    # print(maping)
    fileName = "E:/Facial Feature Motion Clip/rollingInTheDeep.xml"
    videoName = "rollingInTheDeep.mp4"
    extract_landmarks_media_pipe(videoName, "E:/Facial Feature Motion Clip", save_annotated_video=True)
    A[2]
    lm = faceWareLandmarkObject(fileName)
    leftBrow, rightBrow = lm.getBrows()
    nose = lm.getNose()
    plt.plot(leftBrow.mean(axis=1)[:, 1] - nose.mean(axis=1)[:,1])
    plt.show()