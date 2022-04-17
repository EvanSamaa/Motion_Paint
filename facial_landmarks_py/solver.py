import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lmpath", help="path to landmark data output of landmarkUtil.py")
    parser.add_argument("inputpath", help="display a square of a given number")
    parser.add_argument("outputpath", help="display a square of a given number")
    args = parser.parse_args()

    lmPath = "E:/Facial Feature Motion Clip/rollingInTheDeep/raw_mediapipe_landmark.npy"
    lmPath = parser.lmpath
    outputPath = "C:/Users/evansamaa/Desktop/Motion_Paint/facial_landmarks_py/data/motion.json"
    outputPath = parser.outputpath
    inputpath = args.inputpath

    import csv
    with open(inputpath, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        print(', '.join(row))
    spamreader

    # info from video
    t_segment = [0, 168]
    t_ref = 159
    # info of target animation
    start = 0
    fps = 30
    min_cutoff = 0.0001
    beta = 1.5