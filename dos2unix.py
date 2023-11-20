#!/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: python dos2unix.py
"""

import sys

original = "/home/facade/projects/ext/cgna/CGNA_pose_estimation/SMPL/npz_models/basicmodel_m_lbs_10_207_0_v1.0.0.npz"
destination = "/home/facade/projects/ext/cgna/CGNA_pose_estimation/SMPL/npz_models/basicmodel_m_lbs_10_207_0_v1.0.0.npz22"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))
