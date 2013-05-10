# processing of 3D data to generate slices and stuff

import numpy as np

class ObjectModel():
	"""Simpl 3D object as a set of point x,y,z given an a numpy array """
	def __init__(self, points):
		self.points = []
		self.initialize_with_points(points)

	def initialize_with_points(self, points):
		self.points = points

	def cut_clice_curve(self, value, axis):
		# slice through the date and get the boundary curve of the slice plane # assuming non filled volume

		if axis > 2:
			raise Exception("Axis must be only 0=X, 1=Y,2=Z")    

		# get indices where with the value in the given axis
		indices = np.where(self.points[axis,:] == value)

		clen = self.points[axis,:].size
		curve = np.zeros(shape=(2,clen))

		for i in range(0,indices.size):
			curve[0,i] = self.points[0,indices[i]]
			curve[1,i] = self.points[1,indices[i]]

		return curve