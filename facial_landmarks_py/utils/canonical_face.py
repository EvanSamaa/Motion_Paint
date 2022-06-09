from matplotlib import pyplot as plt
import numpy as np

lm_dict = {
  "brow":{
    "rightUpper": [70,63,105,66,107],
    "rightLower": [46,53,52,65,55],
    "leftUpper": [336,296,334,293,300],
    "leftLower": [285,295,282,283,276]
  },
  "nose":{
    "dorsum":[6,197,195,5,4],
    "tipLower":[218,237,44,1,274,457,438],
    "tip":[115,220,45,4,275,440,344]
  },
  "eye": {
    "right": [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],
    "rightUpper": [246,161,160,159,158,157,173],
    "rightLower": [7,163,144,145,153,154,155],
    "rightOuterCorner": [33],
    "rightInnerCorner": [133],
    "left": [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398],
    "leftUpper": [398,384,385,386,387,388,466],
    "leftLower": [382,381,380,374,373,390,249],
    "leftInnerCorner": [362],
    "leftOuterCorner": [263],
    "static": [468,469,470,471,472,473,474,475,476,477]
  },
  "lips": {
    "upperOuter": [185,40,39,37,0,267,269,270,409],
    "upperInner": [191,80,81,82,13,312,311,310,415],
    "lowerOuter": [146,91,181,84,17,314,405,321,375],
    "lowerInner": [95,88,178,87,14,317,402,318,324],
    "outer": [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185],
    "inner": [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]
  },
  "additional_anchors": [127, 356, 132, 361, 33, 133, 362, 263]
}

class ObjLoader(object):
    def __init__(self, fileName):
        self.vertices = []
        self.faces = []
        self.transformed_vertices = []
        self.transformed_faces = []
        ##
        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                    self.vertices.append(vertex)

                elif line[0] == "f":
                    string = line.replace("//", "/")
                    ##
                    i = string.find(" ") + 1
                    face = []
                    for item in range(string.count(" ")):
                        if string.find(" ", i) == -1:
                            face.append(int(string[i:-1].split("/")[0]))
                            break
                        face.append(int(string[i:string.find(" ", i)].split("/")[0]))
                        i = string.find(" ", i) + 1
                    ##
                    self.faces.append(tuple(face))

            f.close()
        except IOError:
            print(".obj file not found.")
        self.vertices = np.array(self.vertices)
        self.faces = np.array(self.faces)
        self.transformed_vertices = np.array(self.vertices)
        self.transformed_faces = np.array(self.faces)

    def project(self, pts):
        # points should be of shape [Number of points, 2]
        self.transformed_vertices = 0
    def transform(self, R, c, t):
        self.transformed_vertices = (c * R @ self.vertices.transpose() + t)
        self.transformed_vertices = self.transformed_vertices.transpose()

    def inside_triangle(self, triangle_idx, pt):
        p0 = self.transformed_vertices[self.faces[triangle_idx][0]]
        p1 = self.transformed_vertices[self.faces[triangle_idx][1]]
        p2 = self.transformed_vertices[self.faces[triangle_idx][2]]
        pt








if __name__ == "__main__":
    face = ObjLoader("../data/canonical_face_model.obj")
    face.transform(np.eye(3), 1, np.zeros((3, 1)))
    print((face.transformed_vertices - face.vertices).max())
    face.project()
