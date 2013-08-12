# processing of 3D data to generate slices and stuff

import numpy as np
import glob


class ObjectModel(object):
    """Simpl 3D object as a set of point x,y,z given an a numpy array """

    def __init__(self):
        self.points = []
        # nothing to do here ;P

    def initialize_with_points(self, points):
        self.points = points

    def cut_clice_curve(self, value, axis):
        # slice through the date and get the boundary curve of the slice plane # assuming non filled volume
        # TODO - fix the buggy issues, using file atm

        if axis > 2:
            raise Exception("Axis must be only 0=X, 1=Y,2=Z")

        # get indices where with the value in the given axis

        # indices = np.where(self.points[axis,:] == value)

        r, c = self.points.shape
        curve = np.zeros(shape=(2, c))

        for i in range(0, c):
            if abs(self.points[2, i] - value) < 0.1:
                print 'Point ', self.points[0, i], self.points[1, i], self.points[2, i]
                curve[0, i] = self.points[0, i]
                curve[1, i] = self.points[1, i]

        # if len(indices) > 0:
        # 	for i in range(0, len(indices)):
        # 		print self.points[0,indices[i]]
        # 		print indices[i]
        # curve[0,i] = self.points[0,indices[i]]
        # curve[1,i] = self.points[1,indices[i]]

        return curve

    def initialize_from_slicefiles(self, dir_path):
        # load 3d data from a collection of slice files
        # only load the list of filenames, read on demand

        self.slice_files = glob.glob(dir_path + "*.txt")

        return self.slice_files

    def load_from_json(self):
        raise("NotImplemented Yet")
