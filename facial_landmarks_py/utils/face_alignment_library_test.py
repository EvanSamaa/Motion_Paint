import face_alignment
from skimage import io
import cv2
from matplotlib import pyplot as plt
def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks[0]):
        pos = (point[0], point[1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 2, color=(0, 255, 255))
    return im


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device="cpu")

input = io.imread('E:/MASC/Motion_paint/facial_landmarking_test/trump.png')[:, :, 0:-1]
print(input.shape) # (480, 699, 3)
preds = fa.get_landmarks(input)
print(preds)
print(preds[0].shape) # (68, 3)
annotated_img = annotate_landmarks(input, preds)

plt.imshow(annotated_img)
plt.show()

# start new
output_string = "1,68,24"
for idx, point in enumerate(preds[0]):
    output_string += "\n"
    output_string += (str(point[0]) + ",")
    output_string += (str(point[1]) + ",")
    output_string += str(point[2])

with open("C:/Users/evansamaa/Desktop/Motion_Paint/motion_transfer/data/trumpExpression.txt", "w") as wf:
    wf.write(output_string)





