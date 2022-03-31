import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
def display_landmark(landmark_arr, fps):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    def update(i):
        label = 'timestep {0}'.format(i)
        print(label)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        landmark_arr_i = landmark_arr[i] + landmark_arr[0]
        fig.clf()
        ax = plt.scatter(landmark_arr_i[:, 0], landmark_arr_i[:, 1])
        return ax
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(landmark_arr)), interval=fps)
    plt.show()
def load_txt(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        meta = lines[0].split(",")
        T = int(meta[0])
        V = int(meta[1])
        fps = int(meta[2])
        outList = []
        for i in range(0, T):
            outNP = np.zeros((V, 3))
            for j in range(0, V):
                line = lines[1+i*V+j].split(",")
                outNP[j, 0] = float(line[0])
                outNP[j, 1] = float(line[1])
                outNP[j, 2] = float(line[2])
            outList.append(outNP)
        return outList
def load_map(file_name):
    map = []
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split(",")
            map.append(int(mapping[1]))
    return map

if __name__ == "__main__":

    lm = np.load()


    lm = np.load("")
    out_landmark = load_txt("E:\\MASC\\Motion_paint\\example_videos\\evanData.txt")
    # display_landmark(out, 30)
    out_blendshape = load_txt("E:\\MASC\\Motion_paint\\JoonHoModel.txt")
    map = load_map("C:\\Users\\evansamaa\\Desktop\\Motion_Paint\\motion_transfer\\data\\valleyGirlMap.txt")
    for i in range(0, len(out_blendshape)):
        out_blendshape[i] = out_blendshape[i][map,:]
    # display_landmark(out, 30)
    print(len(out_blendshape))
    print(out_blendshape[0].shape)