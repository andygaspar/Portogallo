from ctypes import cdll, pointer
import numpy as np

# load the library
lib = cdll.LoadLibrary('./libgeek.so')


# create a Geek class
class Geek(object):

    # constructor
    def __init__(self, vect):
        # attribute
        self.obj = lib.OfferChecker_(vect.ctypes.data, vect.shape[0], vect.shape[1])

        # define method

    def myFunction(self, val):
        lib.Geek_myFunction(self.obj, val)

    def print_value(self):
        lib.Geek_myValue(self.obj)

    def print_vect(self):
        lib.print_vect_(self.obj)

    # create a Geek class object


f = Geek(np.array([[1,2,3,4],[1,1,1,1]]))

# object method calling
f.print_value()
f.print_vect()
f.print_value()