import os

if __name__ == "__main__":
    obj_path = "C:/Users/evansamaa/Desktop/live_link_data/Apple blendshapes51 OBJs/OBJs/"
    filenames = os.listdir(obj_path)
    for f in filenames:
        file_i_obj = str(os.path.join(obj_path, f))
        file_i_new = str(os.path.join(obj_path, f))[:-3] + "gITF"
        print(file_i_obj)
        os.system("obj2gltf -i \"{}\" -o \"{}\"".format(file_i_obj, file_i_new))

