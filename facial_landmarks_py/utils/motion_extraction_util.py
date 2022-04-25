import math
import numpy as np
import json

MIN_CUTOFF = 0.0001
BETA = 5

# this function rotates and translates pointcloud X to match the position of Y
# the algorithm uses X and Y notation, where the input dimension should be (M x N)
# where M is the number of dimension and N is the number of points. the output is
# R, c, t, where they satisfies min_{R, c, t} ||(c*R*X+t) - Y||^2
def compute_rotation(X, Y):
    mu_x = X.mean(axis=1)
    mu_y = Y.mean(axis=1)
    rho2_x = X.var(axis=1).sum()
    rho2_xx = X[0:1].var(axis=1).sum()*3.0
    rho2_xy = X[1:2].var(axis=1).sum()*3.0
    rho2_xz = X[2:3].var(axis=1).sum()*3.0

    rho2_y = Y.var(axis=1).sum()
    cov_xy = 1.0 / X.shape[1] * (Y - np.expand_dims(mu_y, axis=1)) @ (X - np.expand_dims(mu_x, axis=1)).T
    # SVD on the covariance matrix
    U, D, V_T = np.linalg.svd(cov_xy)
    D = np.diag(D)
    # prepare sign flipping matrix S, which need to be altered at some point
    S = np.identity(3)
    # update matrix S based on the rank of cov_xy
    if np.linalg.matrix_rank(cov_xy) >= X.shape[0] - 1:
        if (np.linalg.det(cov_xy) < 0):
            S[-1, -1] = -1
    else:
        det_U = np.linalg.det(U)
        det_V = np.linalg.det(V_T)
        if (det_U * det_V < 0):
            S[-1, -1] = -1
            # compute rotation and scale and translation
    R = U @ S @ V_T
    c = (1.0 / rho2_x) * np.trace(D @ S)
    cc = 1.0 / np.array([rho2_xx, rho2_xy, rho2_xz]) * np.trace(D @ S)
    cc = np.expand_dims(cc, axis=1)
    t = mu_y - cc * R @ mu_x
    # X_prime = c * R @ frame_i.T + np.expand_dims(t, 1)
    # X_prime = rotated_frame_i.T
    return R, cc, np.expand_dims(t, 1)

# input neturalPose should be a numpy array of shape (N, M)
# # data should be a numpy array of shape (T, N, M)
# # staticIndices should be a list of integers, which refers to relavent
# # dimensions in M for the computations
def rotateToNeutral(neutralPose, data, staticIndices):
    outData = np.zeros(data.shape)
    for i in range(0, data.shape[0]):
        frame_t = data[i, staticIndices]
        R, c, t = compute_rotation(frame_t.T, neutralPose[staticIndices].T)
        outData[i] = (c * R @ data[i].T + t).T
    return outData
# get the 6 sheer matrices in three dimensions,
def getSheerMat(dimensions=3):
    out = []
    if dimensions == 3:
        sheer_x_y = np.eye(3)
        sheer_x_y[1, 0] = 1
        sheer_x_z = np.eye(3)
        sheer_x_z[2, 0] = 1

        sheer_y_x = np.eye(3)
        sheer_y_x[0, 1] = 1
        sheer_y_z = np.eye(3)
        sheer_y_z[2, 1] = 1

        sheer_z_x = np.eye(3)
        sheer_z_x[0, 2] = 1
        sheer_z_y = np.eye(3)
        sheer_z_y[1, 2] = 1
        out = [sheer_x_y, sheer_x_z, sheer_y_x, sheer_y_z, sheer_z_x, sheer_z_y]
    return out
# compute a basis for a set of landmarks given the desired tranformation of those landmarks
def shearBasis(landmarks, anchor, transformations):
    # this function expect pointCloud to be in the shape of [3, n]
    # anchor should be the shape [3, 1]
    # return a set of differential blendshapes that represents the pointCloud, centered
    # around the anchor, sheared in all 6 directions.
    # returns a list of arrays, with shape of [3, n], these are differential blendshapes

    out = []
    centered_landmarks = landmarks - anchor
    for trans in transformations:
        out.append((trans @ centered_landmarks + anchor - landmarks))
    return out, landmarks
# compute the weights needed to blend a target set of blendshapes to the target shape
def inverseIK(blendshapes, baseshape, target, indices):
    # baseshape and target should both in shape of [3, n]
    # blendshape is a list of arrays with shape [3, n]

    diff = (target[:, indices] - baseshape[:, indices]).reshape([target.shape[0] * len(indices), 1])
    B = []
    for i in range(0, len(blendshapes)):
        mat = blendshapes[i][:, indices].reshape([target.shape[0] * len(indices), ])
        B.append(np.expand_dims(mat, axis=0))
    B = np.concatenate(B, axis=0)
    w = np.linalg.inv(B @ B.T) @ B @ diff
    return w
# using shear to normalize the data to match a neutral frame
def shearNormalization(data, neutral_frame, shear_landmarkSet, rotation=True, rotation_landmarkset=None):
    data_0 = neutral_frame
    out_data = []
    basis_transformation = getSheerMat()
    for t in range(0, len(data)):
        data_1 = data[t]
        shearCenter = [4]
        if rotation:
            R, c, t = compute_rotation(data_1[rotation_landmarkset].T, data_0[rotation_landmarkset].T)
            print((R @ data_1.T).shape)
            norm_data_1 = (c * R @ data_1.T + t).T
        else:
            norm_data_1 = data_1
        # do the shear transformtions
        basis, neutral = shearBasis(norm_data_1.T, norm_data_1[shearCenter].T, basis_transformation)
        # basis and neutralis that of the entire mesh
        w = inverseIK(basis, neutral, data_0.T, shear_landmarkSet)
        sheared_data_1 = neutral
        for i in range(0, len(basis)):
            sheared_data_1 = sheared_data_1 + w[i] * basis[i]
        sheared_data_1 = sheared_data_1.T
        out_data.append(np.expand_dims(sheared_data_1, axis=0))
    out_data = np.concatenate(out_data, axis=0)
    return out_data
# iteratively shear and rotate the data to match the neutral frame
def iterativeNormalization(data, neutral_frame, rotation_landmarkset, sheer_landmarkset):
    out_data = shearNormalization(data, neutral_frame, sheer_landmarkset, rotation=True,
                                  rotation_landmarkset=rotation_landmarkset)
    out_data = shearNormalization(out_data, neutral_frame, sheer_landmarkset, rotation=True,
                                  rotation_landmarkset=rotation_landmarkset)
    out_data = shearNormalization(out_data, neutral_frame, sheer_landmarkset, rotation=True,
                                  rotation_landmarkset=rotation_landmarkset)
    return out_data

#################################################################################
########################## one euro filter related fns ##########################
#################################################################################
# run one euro filter for a 1D data
def runEuro(t, data):
    out = np.zeros(data.shape)
    out[0] = data[0]
    one_euro_filter = OneEuroFilter(t[0], data[0], min_cutoff=MIN_CUTOFF, beta=BETA)
    for i in range(1, len(t)):
        out[i] = one_euro_filter(t[i], data[i])
    return out
def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)
def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev
        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat
def constrainedOneEuroFilter(data, dataRange, keyFrames):
    # data should be a numpy array of shape (n, )
    # dataRange should be a list of two element, a starting frame and an ending frame [start, end)
    # keyFrames should be a list of keyframes that the model needs is constraint to

    # construct partitions of the signal
    dataPartitions = []
    start = 0
    end = dataRange[1] - dataRange[0]
    for i in range(0, len(keyFrames)):
        kf = keyFrames[i] - start  # conform it to indexing of array
        if i == 0:
            if (kf >= 1):
                dataPartitions.append(data[0:kf])
        else:
            prev_kf = keyFrames[i - 1] - start
            dataPartitions.append(data[prev_kf:kf])
    dataPartitions.append(data[keyFrames[-1] - start:])
    out_dataPartition = []
    # using 1 euro filter to perform changes
    for i in range(0, len(dataPartitions)):
        if i < len(dataPartitions) - 1:
            forward = dataPartitions[i]
            backward = np.flip(dataPartitions[i])

            t = np.arange(0, forward.shape[0])
            alpha = np.arange(forward.shape[0], 0, -1) / forward.shape[0]
            #             plt.plot(backward)
            forward = runEuro(t, forward) * alpha
            backward = np.flip(runEuro(t, backward) * alpha)

            out_dataPartition.append(forward + backward)
        else:
            forward = dataPartitions[i]
            t = np.arange(0, forward.shape[0])
            out_dataPartition.append(runEuro(t, forward))
    return np.concatenate(out_dataPartition)
def outputToFile(path, arr, fps, start):
    # the input should be in the form of a 2D array with shape [n, ]
    arr_length = arr.shape[0]
    dt = 1.0 / fps
    t_arr = np.arange(0, arr_length) * dt + start
    t_arr = t_arr.tolist()
    arr = arr / arr.max()
    v_arr = arr.tolist()

    output = {"t": t_arr, "v": v_arr}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f)
    return

#################################################################################
########################## output to json utils ##########################
#################################################################################
def outputToFile(path, arr, fps, start):
    # the input should be in the form of a 2D array with shape [n, ]
    arr_length = arr.shape[0]
    dt = 1.0 / fps
    t_arr = np.arange(0, arr_length) * dt + start
    t_arr = t_arr.tolist()
    arr = arr / arr.max()
    v_arr = arr.tolist()

    output = {"t": t_arr, "v": v_arr}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f)
    return