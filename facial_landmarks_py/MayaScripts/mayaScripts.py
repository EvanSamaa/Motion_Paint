import maya.cmds as cmds
import json


def textLoading(pathTextFiled):
    thisPath = pathTextFiled

    def loadTxt(*args):
        filepath = cmds.textField(thisPath, text=True, query=True)
        if os.path.exists(filepath):
            # the file is there
            pass
        elif filepath == "":
            filepath = "C:/Users/evansamaa/Desktop/Motion_Paint/facial_landmarks_py/data/motion.json"
        elif os.access(os.path.dirname(filepath), os.W_OK):
            # the file does not exists but write privileges are given
            print
            "invalid"
            return
        else:
            # can not write there with open("filepath")
            print
            "invalid"
            return
        with open(filepath) as f:
            motion = json.load(f)
        t = motion["t"]
        v = motion["v"]
        selected = cmds.ls(sl=True, long=True) or []
        for i in range(0, len(selected)):
            node = selected[i]
            attr = cmds.listAttr(node, keyable=True)[0]
            attr_min, attr_max = cmds.transformLimits(node, query=True, translationY=True)
            for i in range(len(t)):
                cmds.setKeyframe(node + "." + attr, v=v[i], t=t[i])

    return loadTxt

cmds.window(width=300, title="tool")
cmds.columnLayout(adjustableColumn=True)

path = cmds.textField(text="")
cmds.textField(path, edit=True)

# button for loading the motion file
cmds.button(label='loadMotion', command=textLoading(path))
cmds.showWindow()