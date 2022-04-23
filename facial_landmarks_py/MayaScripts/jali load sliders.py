import maya.cmds as cmds
import json


def getSelectedChannels():
    channelBox = mel.eval('global string $gChannelBoxName; $temp=$gChannelBoxName;')  # fetch maya's main channelbox
    attrs = cmds.channelBox(channelBox, q=True, sma=True)
    if not attrs:
        return []
    return attrs


def textLoading(pathTextFiled, intensitySlider):
    thisPath = pathTextFiled
    thisSlider = intensitySlider

    def loadTxt(*args):
        filepath = cmds.textField(thisPath, text=True, query=True)
        if os.path.exists(filepath):
            # the file is there
            pass
        elif filepath == "":
            filepath = "C:/Users/evansamaa/Desktop/Motion_Paint/facial_landmarks_py/data/motion.json"
        elif os.access(os.path.dirname(filepath), os.W_OK):
            # the file does not exists but write privileges are given
            print("invalid")
            return
        else:
            # can not write there with open("filepath")
            print("invalid")
            return
        with open(filepath) as f:
            motion = json.load(f)
        t = motion["t"]
        v = motion["v"]
        selected = cmds.ls(sl=True, long=True) or []
        attrs = getSelectedChannels()
        val = cmds.floatSliderGrp(thisSlider, query=True, value=True)
        print(val)
        for i in range(0, len(attrs)):
            # only one node should have been selected
            node = selected[0]
            attr = attrs[i]
            # attr_min = cmds.attributeQuery(attr, node=node, longName=True)
            # attr_min, attr_max = cmds.transformLimits(node, query=True, translationY=True)
            attr_max = val
            attr_min = 0
            for i in range(len(t)):
                cmds.setKeyframe(node + "." + attr, v=v[i] * attr_max, t=t[i] * 30)

    return loadTxt


def rotationtextLoading(pathTextFiled, intensitySlider):
    thisPath = pathTextFiled
    thisSlider = intensitySlider

    def loadTxt(*args):
        filepath = cmds.textField(thisPath, text=True, query=True)
        if os.path.exists(filepath):
            # the file is there
            pass
        elif filepath == "":
            filepath = "C:/Users/evansamaa/Desktop/Motion_Paint/facial_landmarks_py/data/Rot_motion.json"
        elif os.access(os.path.dirname(filepath), os.W_OK):
            # the file does not exists but write privileges are given
            print("invalid")
            return
        else:
            # can not write there with open("filepath")
            print("invalid")
            return
        with open(filepath) as f:
            motion = json.load(f)
        t = motion["t"]
        v = motion["v"]
        print(v)
        # attr_min = cmds.attributeQuery(attr, node=node, longName=True)
        # attr_min, attr_max = cmds.transformLimits(node, query=True, translationY=True)
        val = cmds.floatSliderGrp(thisSlider, query=True, value=True)
        for i in range(len(t)):
            cmds.setKeyframe("CNT_NECK.rotateZ", v=v[i][0] * val, t=t[i] * 30)
            cmds.setKeyframe("CNT_NECK.rotateX", v=-v[i][2] * val, t=t[i] * 30)
            cmds.setKeyframe("CNT_NECK.rotateY", v=v[i][1] * val, t=t[i] * 30)

    return loadTxt


def clearSelectedNodes(*args):
    selected = cmds.ls(sl=True, long=True) or []
    attrs = getSelectedChannels()
    for i in range(0, len(attrs)):
        node = selected[0]
        attr = attrs[i]
        cmds.cutKey(node, clear=True)
        cmds.setAttr(node + "." + attr, 0)


cmds.window(width=300, title="tool")
cmds.columnLayout(adjustableColumn=True)

path = cmds.textField(text="")
motionIntensity = cmds.floatSliderGrp(label='motion Intensity', field=True, minValue=0, maxValue=10.0, fieldMinValue=0,
                                      fieldMaxValue=10.0, value=6)
cmds.textField(path, edit=True, enterCommand=textLoading(path, motionIntensity))
# button for loading the motion file
cmds.button(label='load Motion', command=textLoading(path, motionIntensity))
rotationIntensity = cmds.floatSliderGrp(label='rotation Intensity', field=True, minValue=0, maxValue=1.0,
                                        fieldMinValue=0, fieldMaxValue=1.0, value=0.5)
cmds.button(label='load Rotation', command=rotationtextLoading(path, rotationIntensity))
cmds.button(label='clear Curves', command=clearSelectedNodes)
cmds.showWindow()

