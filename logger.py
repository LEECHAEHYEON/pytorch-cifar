import datetime
from distutils.version import LooseVersion
import math
import os
import os.path as osp
import shutil

import numpy as np
import pytz
import tqdm


class CustomLogger():
    def __init__(self, out):
        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        # add the learning rate :)
        self.log_headers=['epoch','iteration','train/loss','train/acc','valid/loss','valid/acc','elapsed_time']

        self.timestamp_start = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

    def write(self, is_train, epoch, iteration, loss, acc):

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (datetime.datetime.now(pytz.timezone('Asia/Tokyo'))-self.timestamp_start).total_seconds()
            log = None
            if is_train:
                log = [epoch, iteration] + [loss] + [acc] + [''] * 2 + [elapsed_time]
            else:
                log = [epoch, iteration] + [''] * 2 + [loss] + [acc] + [elapsed_time]

            log = map(str, log)
            f.write(','.join(log) + '\n')


