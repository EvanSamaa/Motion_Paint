# https://github.com/ManuelTS/augmentedFaceMeshIndices/blob/master/Nose.jpg
import moviepy.editor as ed
from tqdm import tqdm
import mediapipe as mp
import dlib
import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import json
import face_alignment

class VideoWriter():
    def __init__(self, opath, fps=25):
        # I don't think this would work with mp4 output, it probably only works with avi
        self.img_array = []
        self.opath = opath
        self.size = (-1, -1)
        self.fps = fps
    def add_frame(self, img):
        self.img_array.append(img)
        height, width, layers = img.shape
        self.size = (width, height)
    def save(self):
        out = cv2.VideoWriter(self.opath, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, self.size)
        for i in range(len(self.img_array)):
            out.write(self.img_array[i])
        out.release()

def get_audio_from_video(file_name, video_folder_path, target_fps = 30, remove=False):
    dir_files = os.listdir(video_folder_path)
    if len(dir_files) == 0:
        print("The directory is empty")
        return []
    video_path = os.path.join(video_folder_path, file_name)
    video_folder = os.path.join(video_folder_path, file_name[:-4])
    try:
        # print(video_folder)
        os.mkdir(video_folder)
    except:
        if remove:
            shutil.rmtree(video_folder, ignore_errors=True)
            os.mkdir(video_folder)
        else:
            return
    my_clip = ed.VideoFileClip(video_path)
    my_clip.audio.write_audiofile(os.path.join(video_folder, "audio.mp3"))
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    meta_data = {}
    if fps <= target_fps:
        meta_data["fps"] = fps
    else:
        factor = fps / target_fps
    meta_data["fps"] = fps
    meta_data["video_path"] = video_path
    meta_data["audio_path"] = os.path.join(video_folder, "audio.mp3")
    with open(os.path.join(video_folder, "other_info.json"), 'w') as outfile:
        json.dump(meta_data, outfile)
def split_video_to_images(file_name, video_folder_path, target_fps = 30, remove=False):
    # filename can just be the name of the file,
    # the video must be in the video folder_path
    frames = []
    dir_files = os.listdir(video_folder_path)
    if len(dir_files) == 0:
        print("The directory is empty")
        return []

    for video in os.listdir(video_folder_path):
        # print(video)
        if video == file_name:
            video_path = os.path.join(video_folder_path, video)
            video_folder = os.path.join(video_folder_path, video[:-4])
            try:
                # print(video_folder)
                os.mkdir(video_folder)
            except:
                if remove:
                    shutil.rmtree(video_folder, ignore_errors=True)
                    os.mkdir(video_folder)
                else:
                    dir_ls = os.listdir(video_folder)
                    counter = 0
                    for i in range(0, len(dir_ls)):
                        if dir_ls[i][-4:] == ".jpg":
                            frames.append(video_folder + "/frame%d.jpg" % counter)
                            counter = counter + 1
                    print("video to image conversion was done before, {} frames are loaded".format(len(frames)))
                    return frames
            my_clip = ed.VideoFileClip(video_path)
            my_clip.audio.write_audiofile(os.path.join(video_folder, "audio.mp3"))
            vidcap = cv2.VideoCapture(video_path)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            meta_data = {}
            if fps <= target_fps:
                meta_data["fps"] = fps
            else:
                factor = fps/target_fps
            meta_data["fps"] = fps
            meta_data["video_path"] = video_path
            meta_data["audio_path"] = os.path.join(video_folder, "audio.mp3")
            with open(os.path.join(video_folder, "other_info.json"), 'w') as outfile:
                json.dump(meta_data, outfile)
            success, image = vidcap.read()
            count = 0
            while success:
                cv2.imwrite(video_folder + "/frame%d.jpg" % count, image)  # save frame as JPEG file
                success, image = vidcap.read()
                frames.append(video_folder + "/frame%d.jpg" % count)
                count += 1
    print("video to image conversion done")
    return frames
def get_wav_from_video(file_name, video_folder_path):
    dir_files = os.listdir(video_folder_path)
    if len(dir_files) == 0:
        print("The directory is empty")
        return []
    for video in os.listdir(video_folder_path):
        # print(video)
        if video == file_name:
            video_path = os.path.join(video_folder_path, video)
            my_clip = ed.VideoFileClip(video_path)
            my_clip.audio.write_audiofile(video_path[:-3] + "wav")
    return video_path[:-3] + "wav"
def mp32wav(file_name, audio_folder_path):
    dir_files = os.listdir(audio_folder_path)
    if len(dir_files) == 0:
        print("The directory is empty")
        return []
    for audio in os.listdir(audio_folder_path):
        if audio == file_name:
            video_path = os.path.join(audio_folder_path, audio)
            music, sr = librosa.load(os.path.join(audio_folder_path, audio))
            # librosa.output.write_wav(video_path[:-3] + "wav", music)
            # music.write(video_path[:-3] + "wav")
            write(video_path[:-3] + "wav", sr, music)
def align2clips(clip1, clip2):
    # clip1 should be the shorter clip
    diff = clip2.shape[0] - clip1.shape[0]
    min_val = np.inf
    min_index = -1
    for i in range(0, diff):
        temp_aligned_clip2 = clip2[i:i + clip1.shape[0]]
        val = np.linalg.norm(temp_aligned_clip2 - clip1)
        if val <= min_val:
            min_val = val
            min_index = i
    return clip2[min_index:min_index + clip1.shape[0]]
def extract_landmarks_Fan(input_video, input_dir, show_annotated_video = False, show_normalized_pts = False, save_annotated_video = False,  tolerance = 0.01):

    # input_video should just be the name of the file, not the absolute path
    # input_dir would be the absolute path to the folder containing the input video.
    # the processed data would be in a folder with input_video[-4:] as the title, which includes a numpy file, as well
    # other files such as a json file for the metadata of the video, as well as the audio component of the video.

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    output_path = os.path.join(os.path.join(input_dir, input_video[:-4]), "2D_FAN_landmark.npy")
    raw_output_path = os.path.join(os.path.join(input_dir, input_video[:-4]), "raw_FAN_landmark.npy")
    get_audio_from_video(input_video, input_dir,)
    # set up cv2 object for querying images from video
    cap = cv2.VideoCapture(os.path.join(os.path.join(input_dir, input_video)))
    count = 0


    with open(os.path.join(input_dir, input_video[:-4] + "/other_info.json")) as json_file:
        metadata = json.load(json_file)
    fps = metadata["fps"]
    if save_annotated_video:
        vr = VideoWriter(os.path.join(input_dir, input_video[:-4] + "FAN_labeled.avi"), fps=fps)

    landmark_output = []
    raw_landmark_output = []
    pbar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device="cpu")
    # for idx, file in enumerate(IMAGE_FILES):
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        # Convert the BGR image to RGB before processing.

        land_mark_matrix_pts = fa.get_landmarks(image)

        # Print and draw face mesh landmarks on the image.
        if len(land_mark_matrix_pts) == 0:
            landmark_output.append(np.zeros((68, 2)))
            raw_landmark_output.append(np.zeros((68, 3)))
            continue
        land_mark_matrix_pts = land_mark_matrix_pts[0]
        # https://github.com/ManuelTS/augmentedFaceMeshIndices/blob/master/Nose.jpg points of the face model

        plane_pts = [land_mark_matrix_pts[31], land_mark_matrix_pts[33], land_mark_matrix_pts[35]]
        # rotate the projected matrix to face the camerra
        n = np.cross(plane_pts[2] - plane_pts[1], plane_pts[0] - plane_pts[1])
        n = n / np.linalg.norm(n)
        R = rotation_matrix_from_vectors(n, np.array([0, 0, 1]))
        rotated_land_marks = np.expand_dims(land_mark_matrix_pts, axis=2)
        R = np.expand_dims(R, axis=0)
        rotated_land_marks = R @ rotated_land_marks
        projected_land_marks = rotated_land_marks[:, 0:2, 0]
        projected_land_marks = projected_land_marks - projected_land_marks[4]
        # rotate the face again so eyes are parallel to the screen
        nose_ridge_vector = (projected_land_marks[6, :])
        nose_ridge_vector = nose_ridge_vector / np.linalg.norm(nose_ridge_vector)
        target_nose_ridge_direction = np.array([0, 1])
        abs_angle_diff = np.arccos(np.dot(nose_ridge_vector, target_nose_ridge_direction))
        theta = abs_angle_diff
        r = np.array(((np.cos(theta), -np.sin(theta)),
                      (np.sin(theta), np.cos(theta))))
        diff = np.linalg.norm(r @ nose_ridge_vector - target_nose_ridge_direction)
        if diff >= tolerance:
            theta = - theta
            r = np.array(((np.cos(theta), -np.sin(theta)),
                          (np.sin(theta), np.cos(theta))))
            if np.linalg.norm(r @ nose_ridge_vector - target_nose_ridge_direction) >= diff:
                theta = - theta
                r = np.array(((np.cos(theta), -np.sin(theta)),
                              (np.sin(theta), np.cos(theta))))
        normalized_landmark = np.expand_dims(r, axis=0) @ np.expand_dims(projected_land_marks, axis=2)
        landmark_output.append(normalized_landmark[:, :, 0])
        raw_landmark_output.append(land_mark_matrix_pts)
        pbar.update(1)
    pbar.close()
    np.save(raw_output_path, raw_landmark_output)
    np.save(output_path, landmark_output)
    output_string = "{},68,{}".format(len(landmark_output), int(fps))
    for i in range(0, len(raw_landmark_output)):
        for k in range(0, raw_landmark_output[i].shape[0]):
            output_string += "\n"
            output_string += (str(raw_landmark_output[i][k, 0]) + ",")
            output_string += (str(raw_landmark_output[i][k, 1]) + ",")
            output_string += str(raw_landmark_output[i][k, 2])
    with open("E:\\MASC\\Motion_paint\\example_videos\\evanData.txt", "w") as wf:
        wf.write(output_string)


    return output_path
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
def extract_landmarks_media_pipe(input_video, input_dir, show_annotated_video = False, show_normalized_pts = False, save_annotated_video = False,  tolerance = 0.01):

    # input_video should just be the name of the file, not the absolute path
    # input_dir would be the absolute path to the folder containing the input video.
    # the processed data would be in a folder with input_video[-4:] as the title, which includes a numpy file, as well
    # other files such as a json file for the metadata of the video, as well as the audio component of the video.

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    output_path = os.path.join(os.path.join(input_dir, input_video[:-4]), "2D_mediapipe_landmark.npy")
    raw_output_path = os.path.join(os.path.join(input_dir, input_video[:-4]), "raw_mediapipe_landmark.npy")
    # if os.path.exists(output_path):
    #     return output_path

    get_audio_from_video(input_video, input_dir,)

    # set up cv2 object for querying images from video
    cap = cv2.VideoCapture(os.path.join(os.path.join(input_dir, input_video)))
    count = 0


    with open(os.path.join(input_dir, input_video[:-4] + "/other_info.json")) as json_file:
        metadata = json.load(json_file)
    fps = metadata["fps"]
    if save_annotated_video:
        vr = VideoWriter(os.path.join(input_dir, input_video[:-4] + "mediapipe_labeled.avi"), fps=fps)

    landmark_output = []
    raw_landmark_output = []
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            refine_landmarks=True) as face_mesh:
        pbar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # for idx, file in enumerate(IMAGE_FILES):
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Print and draw face mesh landmarks on the image.\
            if not results.multi_face_landmarks:
                landmark_output.append(np.zeros((478, 2)))
                raw_landmark_output.append(np.zeros((478, 3)))
                continue
            # https://github.com/ManuelTS/augmentedFaceMeshIndices/blob/master/Nose.jpg points of the face model
            # print(results.multi_face_landmarks)
            face_landmarks = results.multi_face_landmarks[0].landmark
            land_mark_matrix_pts = np.zeros((478, 3))
            for i in range(0, len(face_landmarks)):
                land_mark_matrix_pts[i, 0] = face_landmarks[i].x
                land_mark_matrix_pts[i, 1] = face_landmarks[i].y
                land_mark_matrix_pts[i, 2] = face_landmarks[i].z

            plane_pts = [land_mark_matrix_pts[98], land_mark_matrix_pts[327], land_mark_matrix_pts[168]]
            # rotate the projected matrix to face the camerra
            n = np.cross(plane_pts[2] - plane_pts[1], plane_pts[0] - plane_pts[1])
            n = n / np.linalg.norm(n)
            R = rotation_matrix_from_vectors(n, np.array([0, 0, 1]))
            rotated_land_marks = np.expand_dims(land_mark_matrix_pts, axis=2)
            R = np.expand_dims(R, axis=0)
            rotated_land_marks = R @ rotated_land_marks
            projected_land_marks = rotated_land_marks[:, 0:2, 0]
            projected_land_marks = projected_land_marks - projected_land_marks[4]
            # rotate the face again so eyes are parallel to the screen
            nose_ridge_vector = (projected_land_marks[6, :])
            nose_ridge_vector = nose_ridge_vector / np.linalg.norm(nose_ridge_vector)
            target_nose_ridge_direction = np.array([0, 1])
            abs_angle_diff = np.arccos(np.dot(nose_ridge_vector, target_nose_ridge_direction))
            theta = abs_angle_diff
            r = np.array(((np.cos(theta), -np.sin(theta)),
                          (np.sin(theta), np.cos(theta))))
            diff = np.linalg.norm(r @ nose_ridge_vector - target_nose_ridge_direction)
            if diff >= tolerance:
                theta = - theta
                r = np.array(((np.cos(theta), -np.sin(theta)),
                              (np.sin(theta), np.cos(theta))))
                if np.linalg.norm(r @ nose_ridge_vector - target_nose_ridge_direction) >= diff:
                    theta = - theta
                    r = np.array(((np.cos(theta), -np.sin(theta)),
                                  (np.sin(theta), np.cos(theta))))
            normalized_landmark = np.expand_dims(r, axis=0) @ np.expand_dims(projected_land_marks, axis=2)
            landmark_output.append(normalized_landmark[:, :, 0])
            raw_landmark_output.append(land_mark_matrix_pts)
            if show_normalized_pts:
                # plt.subplot(2,1,1)
                plt.scatter(normalized_landmark[:, 0], normalized_landmark[:, 1])
                plt.scatter(normalized_landmark[4, 0], normalized_landmark[4, 1])
                plt.scatter(normalized_landmark[98, 0], normalized_landmark[98, 1])
                plt.scatter(normalized_landmark[327, 0], normalized_landmark[327, 1])
                # plt.show()
                plt.show(block=False)
                plt.pause(0.01)
                plt.close()
                # annotate the image
            if show_annotated_video or save_annotated_video:
                annotated_image = image.copy()
                for face_landmarks in results.multi_face_landmarks:
                    # print('face_landmarks:', face_landmarks)
                    # mp_drawing.draw_detection()
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                # imgs_arr.append(annotated_image)
                # cv2.imwrite('./tmp/annotated_image' + str(idx) + '.png', annotated_image)
                if show_annotated_video:
                    cv2.imshow("k", annotated_image)
                    cv2.waitKey(1)
                if save_annotated_video:
                    vr.add_frame(annotated_image)
            pbar.update(1)
        pbar.close()
        if save_annotated_video:
            vr.save()
        landmark_output = np.array(landmark_output)
        raw_landmark_output = np.array(raw_landmark_output)
        np.save(raw_output_path, raw_landmark_output)
        np.save(output_path, landmark_output)
        return output_path
def normalize_open_cv_face(facearr, h, w, tolerance):
    # normalize to the center of the graph
    facearr = facearr - facearr[33]
    facearr = facearr / np.array([[w, h]])
    nose_ridge_vector = (facearr[28, :])
    nose_ridge_vector = nose_ridge_vector / np.linalg.norm(nose_ridge_vector)
    target_nose_ridge_direction = np.array([0, 1])
    abs_angle_diff = np.arccos(np.dot(nose_ridge_vector, target_nose_ridge_direction))
    theta = abs_angle_diff
    r = np.array(((np.cos(theta), -np.sin(theta)),
                  (np.sin(theta), np.cos(theta))))
    diff = np.linalg.norm(r @ nose_ridge_vector - target_nose_ridge_direction)
    if diff >= tolerance:
        theta = - theta
        r = np.array(((np.cos(theta), -np.sin(theta)),
                      (np.sin(theta), np.cos(theta))))
        if np.linalg.norm(r @ nose_ridge_vector - target_nose_ridge_direction) >= diff:
            theta = - theta
            r = np.array(((np.cos(theta), -np.sin(theta)),
                          (np.sin(theta), np.cos(theta))))

    normalized_landmark = np.expand_dims(r, axis=0) @ np.expand_dims(facearr, axis=2)
    return normalized_landmark
def extract_landmarks_opencv(input_video, input_dir, show_annotated_video = False, show_normalized_pts = False, save_annotated_video = False,  tolerance = 0.01):

    output_path = os.path.join(os.path.join(input_dir, input_video[:-4]), "cv_landmark.npy")
    # preparation of the models
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")
    detector = dlib.get_frontal_face_detector()
    # split video into images for the pipeline
    img_list = split_video_to_images(input_video,
                                        input_dir)
    with open(os.path.join(input_dir, input_video[:-4] + "/other_info.json")) as json_file:
        metadata = json.load(json_file)
    fps = metadata["fps"]
    if save_annotated_video:
        vr = VideoWriter(os.path.join(input_dir, input_video[:-4] + "cv_labeled.avi"), fps=fps)
    normalized_landmarks = []
    pbar = tqdm(total=len(img_list))
    for source_img in img_list:
        img = cv2.imread(source_img)
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0:
            normalized_landmarks.append(np.zeros((68, 2)))
        else:
            face = faces[0]
            h = abs(face.top() - face.bottom())
            w = abs(face.left() - face.right())
            face_arr = np.zeros((68, 2))
            landmarks = predictor(image=gray, box=face)
            for n in range(0, 68):
                face_arr[n, 0] = landmarks.part(n).x
                face_arr[n, 1] = landmarks.part(n).y
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
            # normalize the landmarks so they center around the nose, and that the eyes are level
            face_arr = normalize_open_cv_face(face_arr, h, w, tolerance)
            normalized_landmarks.append(face_arr[:, :, 0])
            if show_normalized_pts:
                # plt.subplot(2,1,1)
                plt.scatter(face_arr[:, 0], face_arr[:, 1])
                # plt.show()
                plt.show(block=False)
                plt.pause(0.01)
                plt.close()
        pbar.update(1)
        # show the image
        if show_annotated_video:
            imS = cv2.resize(img, (960, int(1080 * (960 / 1920))))
            cv2.imshow(winname="Face", mat=imS)
            cv2.waitKey(delay=1)
        # save the image
        if save_annotated_video:
            vr.add_frame(img)
    pbar.close()
    if save_annotated_video:
        vr.save()
    landmark_output = np.array(normalized_landmarks)
    np.save(output_path, landmark_output)
def extract_landmark_media_pipe_single_image(file_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    landmark_output = []
    tolerance = 0.001
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
        imgs_arr = []
        image = cv2.imread(file)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        face_landmarks = results.multi_face_landmarks[0].landmark
        land_mark_matrix_pts = np.zeros((468, 3))
        for i in range(0, len(face_landmarks)):
            land_mark_matrix_pts[i, 0] = face_landmarks[i].x
            land_mark_matrix_pts[i, 1] = face_landmarks[i].y
            land_mark_matrix_pts[i, 2] = face_landmarks[i].z
        plane_pts = [land_mark_matrix_pts[98], land_mark_matrix_pts[327], land_mark_matrix_pts[168]]
        # rotate the projected matrix to face the camerra
        n = np.cross(plane_pts[2] - plane_pts[1], plane_pts[0] - plane_pts[1])
        n = n / np.linalg.norm(n)
        R = rotation_matrix_from_vectors(n, np.array([0, 0, 1]))
        rotated_land_marks = np.expand_dims(land_mark_matrix_pts, axis=2)
        R = np.expand_dims(R, axis=0)
        rotated_land_marks = R @ rotated_land_marks
        projected_land_marks = rotated_land_marks[:, 0:2, 0]
        projected_land_marks = projected_land_marks - projected_land_marks[4]

        nose_ridge_vector = (projected_land_marks[6, :])
        nose_ridge_vector = nose_ridge_vector / np.linalg.norm(nose_ridge_vector)
        target_nose_ridge_direction = np.array([0, 1])
        abs_angle_diff = np.arccos(np.dot(nose_ridge_vector, target_nose_ridge_direction))
        theta = abs_angle_diff
        r = np.array(((np.cos(theta), -np.sin(theta)),
                      (np.sin(theta), np.cos(theta))))
        diff = np.linalg.norm(r @ nose_ridge_vector - target_nose_ridge_direction)
        if diff >= tolerance:
            theta = - theta
            r = np.array(((np.cos(theta), -np.sin(theta)),
                          (np.sin(theta), np.cos(theta))))
            if np.linalg.norm(r @ nose_ridge_vector - target_nose_ridge_direction) >= diff:
                theta = - theta
                r = np.array(((np.cos(theta), -np.sin(theta)),
                              (np.sin(theta), np.cos(theta))))

        normalized_landmark = np.expand_dims(r, axis=0) @ np.expand_dims(projected_land_marks, axis=2)
        landmark_output = normalized_landmark[:, :, 0]
        return landmark_output

if __name__ == "__main__":

    extract_landmarks_media_pipe("rollingInTheDeep.mp4",
                                 "E:/Facial Feature Motion Clip", save_annotated_video=True)

    # extract_landmarks_Fan("EvanGoPro.mp4",
    #                              "E:\\MASC\\Motion_paint\\example_videos", save_annotated_video=False, show_normalized_pts=True)
    A[2]
    show_annotated_video = False
    show_normalized_pts = False
    tolerance = 0.01

    video_title = ["video.mp4", "video.mp4", "video.mp4", "video.mp4", "video.mp4"]
    video_path = ["E:/facial_data_analysis_videos/1", "E:/facial_data_analysis_videos/2", "E:/facial_data_analysis_videos/3", "E:/facial_data_analysis_videos/4", "E:/facial_data_analysis_videos/5"]
    for i in range(0, 5):
        extract_landmarks_media_pipe(video_title[i],
                                 video_path[i], save_annotated_video=False)
    # extract_landmarks_opencv(video_title[0],
    #                          video_path[0], save_annotated_video=True)

