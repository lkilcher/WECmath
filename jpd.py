from .base2 import tp_edges, hs_edges
import numpy as np
import powermatrix as pmat


time_factors = {'Y': 365 * 24,
                'M': 30 * 24,
                'D': 24,
                'H': 1,
                'm': 1. / 60,
                's': 1. / 3600, }
tmpval = time_factors['s']
for nm in ['ms', 'us', 'ns', 'ps', 'fs', 'as']:
    tmpval *= 1e-3
    time_factors[nm] = tmpval
del tmpval


def _td2hour(td):
    dtn = td.dtype.name
    if not dtn.startswith('timedelta64'):
        raise Exception("Wrong data type for 'td2hour' function.")
    dtn = dtn[12:-1]  # This strips 'timedelta64[' and ']'
    return td.astype(int, subok=False) * time_factors[dtn]


class TimeSeries(object):

    def __init__(self, data=None, time_edges=None, ):
        self.data = data
        dtp = time_edges.dtype.name
        self.time_edges = time_edges
        if dtp.startswith('datetime64') and dtp[-2] in ['Y', 'M']:
            time_edges = time_edges.astype('datetime64[D]')
        self._Nhour = _td2hour(np.diff(time_edges))
        self.time = time_edges[:-1] + np.diff(time_edges) / 2.0

    def masked(self, fractime_thresh=0.8):
        return np.ma.masked_where(self.fractime < fractime_thresh,
                                  self.data, copy=False)

    def reshape(self, s, pad=0):
        out = PowerTimeSeries(self)
        for nm in ['time', 'fractime', '_Nhour', 'data']:
            if hasattr(self, nm):
                setattr(out, nm, _reshape(getattr(self, nm), s, pad))
        return out


class ResMat(TimeSeries):
    """The 'Resource Matrix' is the number of hours at each sea state
    (Hs, Tp).

    This is essentially the joint probability distribution, but it's
    units are hours (i.e. it's a histogram, not a distribution/density
    function).
    """

    @property
    def fractime(self, ):
        return self.data.sum(-1).sum(-1) / self._Nhour

    def __mul__(self, other):
        if isinstance(other, pmat.PowerMatrix):
            out = PowerTimeSeries(self)
            out.data = (self.data * other.data[None]).sum(-1).sum(-1)
            out.wec_rated_power = other.rated_power
            return out
        raise TypeError("unsupported operand type(s) for *: {} and {}"
                        .format(self.__class__, other.__class__))
    __rmul__ = __mul__

    @property
    def average(self, ):
        return (self.data / self._Nhour[:, None, None]).mean(0)

    @property
    def total_years(self, ):
        return self.data.sum() / 365. / 24.


def calc_resmat(hs, tp, time, time_edges):
    dt = np.median(_td2hour(np.diff(time)))
    if time.dtype > time_edges.dtype:
        te = time_edges.astype(time.dtype)
    elif time_edges.dtype > time.dtype:
        time = time.astype(time_edges.dtype)
    time = time.astype(int)
    te = te.astype(int)
    mat, bins = np.histogramdd(np.stack((time, hs, tp), 1),
                               [te, hs_edges, tp_edges])
    mat *= dt
    out = ResMat(mat, time_edges)
    return out


def _reshape(arr, s, pad=0):
    shape = list(arr.shape)
    if isinstance(s, int):
        s = [-1, s]
    shape = s + shape[1:]
    out = arr.reshape(shape)
    if isinstance(pad, int):
        pad = [pad, pad]
    pad = list(pad)
    if not pad == [0, 0]:
        p0, pe = pad
        stack = []
        if p0 > 0:
            stack += [np.vstack((np.zeros(p0, dtype=arr.dtype)[None], out[:-1, -p0:]))]
        stack += [out]
        if pe > 0:
            stack += [np.vstack((out[1:, :pe], np.zeros(pe, dtype=arr.dtype)[None]))]
        out = np.hstack(stack)
        # mask = np.zeros(out.shape, dtype='bool')
        # mask[0, 0] = True
        # mask[-1, -1] = True
        # out = np.ma.masked_where(mask, out, copy=False)
    return out


class PowerTimeSeries(TimeSeries):

    def __init__(self, resmat):
        self.time_edges = resmat.time_edges
        self.time = resmat.time
        self.fractime = resmat.fractime
        self._Nhour = resmat._Nhour
        try:
            self.wec_rated_power = resmat.wec_rated_power
        except:
            pass

    def __repr__(self, ):
        return self.masked().__str__()

    @property
    def capacity(self, ):
        return self._Nhour * self.wec_rated_power

    def cap_factor(self, fractime_thresh=None):
        if fractime_thresh is None:
            return self.data / self.capacity
        else:
            return self.masked(fractime_thresh) / self.capacity


if __name__ == '__main__':
    import mhk.cdip.base as cdip

    if 'dat' not in vars():
        dat = cdip.get_thredd(139)

        hs = dat.ncdf.variables['waveHs']
        tp = dat.ncdf.variables['waveTp']
        time = dat.waveTime

        timeout = np.arange(np.datetime64('2006-01'),
                            np.datetime64('2016-11')).astype('datetime64[D]')

    mat = calc_resmat(hs, tp, time, timeout)

    pow = pmat.pelamis * mat
