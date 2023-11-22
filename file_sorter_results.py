# -*- coding: utf-8 -*-
#OS and General
import os
import time
import numpy as np

source_fp = '/beegfs/car/njm/OUTPUT/results/'
files = os.listdir(source_fp)

output_fp = '/beegfs/car/njm/OUTPUT/results/'
output_files = os.listdir(output_fp)

matched = [s for s in output_files if any(xs in s for xs in files)]

for match in matched:
    time_since = time.time() - os.stat(output_fp + match).st_mtime
    if time_since > 1800:
            print(match[:-4])
            os.path.isfile(source_fp + match[:-4])





