import os
from aspose.threed import Scene, FileFormat

if __name__ == "__main__":
    file_path = "C:/Users/evansamaa/Desktop/live_link_data/LiveLinkFace_20220926_MySlate_7_Evans_Iphone/MySlate_7_Evans_Iphone.csv"
    obj_path = "C:/Users/evansamaa/Desktop/live_link_data/Apple blendshapes51 OBJs/OBJs/"
    filenames = os.listdir(obj_path)
    for f in filenames:
        # Load the FBX in an object of Scene
        fbxtoObj = Scene(str(os.path.join(obj_path, f)))
        # save FBX as a OBJ
        fbxtoObj.save(str(os.path.join(obj_path, f))[:-3]+"gITF", FileFormat.GLTF);
