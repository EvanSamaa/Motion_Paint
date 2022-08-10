import json
import maya.OpenMaya as OpenMaya
import json
import math

def select_metahuman_landmark():
    landmark_data = json.load(open("C:/Users/evansamaa/Desktop/Motion_Paint/facial_landmarks_py/data/metahumanLOD1_landmark_vertex_mapping.json"))
    indexes = landmark_data["index"]
    cmds.select(clear=True)
    for index in indexes:
        cmds.select("head_lod1_meshShape.vtx[{}]".format(index), add=True)
        
    
def particleFillSelection(indices):
   # this function simply return the vertex position of the given indices as 
   # indicated by indices parameter
	# get the active selection
	selection = OpenMaya.MSelectionList()
	OpenMaya.MGlobal.getActiveSelectionList( selection )
	iterSel = OpenMaya.MItSelectionList(selection, OpenMaya.MFn.kMesh)
	# go througt selection
	while not iterSel.isDone():
		# get dagPath
		dagPath = OpenMaya.MDagPath()
		iterSel.getDagPath( dagPath )
		# create empty point array
		inMeshMPointArray = OpenMaya.MPointArray()
		# create function set and get points in world space
		currentInMeshMFnMesh = OpenMaya.MFnMesh(dagPath)
		currentInMeshMFnMesh.getPoints(inMeshMPointArray, OpenMaya.MSpace.kWorld)
		# put each point to a list
		pointList = []

		for i in range( len(indices) ) :
 
			pointList.append( [inMeshMPointArray[indices[i]][0], inMeshMPointArray[indices[i]][1], inMeshMPointArray[indices[i]][2]] )
 
		return pointList

def export_metahuman_landmark_over_time(start, end, output_path):
    out = {}
    landmark_data = json.load(open("C:/Users/evansamaa/Desktop/Motion_Paint/facial_landmarks_py/data/metahumanLOD1_landmark_vertex_mapping.json"))
    indexes = landmark_data["index"]
    for i in range(start, end):
        cmds.currentTime(i, edit=True)
        out[i] = particleFillSelection(indexes)
        
    with open(output_path, "w") as outfile:
            json.dump(out, outfile)

export_metahuman_landmark_over_time(0, 500, "C:/Users/evansamaa/Desktop/Motion_Paint/facial_landmarks_py/data/output_key_frames.json")
