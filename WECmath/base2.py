import numpy as np
import os

pkg_dir = os.path.dirname(os.path.realpath(__file__))

hs_edges = np.arange(0, 11.1, 0.25)
tp_edges = np.arange(2, 25.1, 0.25)

hs_centers = hs_edges[:-1] + np.diff(hs_edges) / 2
tp_centers = tp_edges[:-1] + np.diff(tp_edges) / 2
