import numpy as np


def centers2edges(arr):
    d = np.diff(arr)
    outarr = np.empty(len(arr) + 1, dtype=arr.dtype)
    outarr[1:-1] = arr[:-1] + d / 2
    outarr[0] = arr[0] - d[0] / 2
    outarr[-1] = arr[-1] + d[-1] / 2
    return outarr


# class powermatrix(object):

#     def __init__(self, mat, period, Hs, name=None):
#         self.period = period
#         self.mat = mat
#         self.Hs = Hs
#         self.name = name

#     def __repr__(self, ):
#         outstr = '<'
#         if self.name is not None:
#             outstr += "'{}' ".format(self.name)
#         outstr += ('Power Matrix\n  ({} periods from {} to {} s)'
#                    .format(len(self.period),
#                            self.period[0],
#                            self.period[-1]))
#         outstr += ('\n  ({} Hs from {} to {} m)'
#                    .format(len(self.Hs), self.Hs[0], self.Hs[-1]))
#         outstr += '>'
#         return outstr

#     @property
#     def period_edges(self, ):
#         return centers2edges(self.period)

#     @property
#     def Hs_edges(self, ):
#         return centers2edges(self.Hs)
