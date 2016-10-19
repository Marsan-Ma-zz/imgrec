from __future__ import division, print_function, absolute_import

from lib import data_util
from lib.googlenet import GoogLeNet

import pickle, gzip
import numpy as np
import tflearn.datasets.oxflower17 as oxflower17


#-------------------------------
#   Training
#-------------------------------
# scope_name, label_size = '17flowers', 17
scope_name, label_size = '17portraits', 9

gnet = GoogLeNet(img_size=227, label_size=label_size, gpu_memory_fraction=1.0, scope_name=scope_name)
down_sampling = {str(n): 10000 for n in range(13)}
pkl_files = gnet.get_data(dirname=scope_name, down_sampling=down_sampling)

for f in pkl_files:
    X, Y = pickle.load(gzip.open(f, 'rb'))
    gnet.fit(X, Y, n_epoch=10)
