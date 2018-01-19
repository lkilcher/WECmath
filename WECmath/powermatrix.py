import numpy as np
import base2 as base
from scipy.interpolate import interp2d
import pkg_resources


class PowerMatrix(object):
    """The power matrix class.

    This class holds the power matrix, which is the amount of power
    (in kW) that a device produces for a given sea state (Hs, Tp).
    """

    def __init__(self, mat=None, name=None):
        self.Tp = base.tp_centers.copy()
        self.Hs = base.tp_centers.copy()
        self.data = mat
        self.name = name

    def __repr__(self, ):
        outstr = '<'
        if self.name is not None:
            outstr += "'{}'".format(self.name)
        else:
            outstr += "!?UNNAMED?!"
        outstr += ' Power Matrix ({} kW)>'.format(self.rated_power)
        return outstr

    @property
    def rated_power(self, ):
        return self.data.max()


def _checkpad(vec, range):
    """This function pads a vector to zero if it is inside the limits
    of range."""
    pad = [0, 0]
    if vec[0] > min(range):
        vec = np.concatenate((vec[0] - np.diff(vec[:2]) / 2, vec))
        pad[0] = 1
    if vec[-1] < max(range):
        vec = np.concatenate((vec, vec[-1] + np.diff(vec[-2:]) / 2))
        pad[-1] = 1
    return vec, pad


def interp2pmat(pmat, hs, tp, name=None):

    hs, pad_hs = _checkpad(hs, base.hs_centers)
    tp, pad_tp = _checkpad(tp, base.tp_centers)
    pmat = np.pad(pmat, [pad_hs, pad_tp],
                  mode='constant', constant_values=0)
    func = interp2d(tp, hs, pmat, fill_value=0)
    return PowerMatrix(func(base.tp_centers, base.hs_centers), name=name)


tmpdat = np.genfromtxt(pkg_resources.resource_filename('WECmath',
                                                       'data/pelamis.csv'),
                       delimiter=',')
pelamis = interp2pmat(pmat=tmpdat[1:, 1:],
                      hs=tmpdat[1:, 0],
                      tp=tmpdat[0, 1:],
                      name='Pelamis')
del tmpdat
