#!/usr/bin/env python3

"""Make CASToR compatible normalisation file from LORs with crystal pairs at least MINANG apart.

Usage: make_normalisation.py (--geom GEOMFILE) [--ang MINANG] [--atn] OUTPUT

Arguments:
    OUTPUT binary filename with normalisation info

Options:
    --geom=GEOMFILE File with the geometry to be used for generating
                    crystal indices.
    --ang=MINANG    Minimum transverse angular separation of crystals to be
                    processed in degrees.
                    [default: 90]
    --atn           Use attenuation correction factors.
"""

from math import fabs
import struct

import numpy as np

from docopt import docopt
from numba  import jit

from src.geometry import attenuation_correction
from src.geometry import get_geometry_histogram


@jit
def generate_normalisation_lors(outfile, geom_arr, atn):
    ncryst = len(geom_arr)

    binOut = open(outfile, 'wb')
    lor_count = 0
    for i, pos1 in enumerate(geom_arr[:-1]):
        for j, pos2 in enumerate(geom_arr[i+1:]):
            angDiff   = fabs(pos1[1] - pos2[1])
            angSep    = min(angDiff, 2 * np.pi - angDiff)
            if angSep < ang_sep: continue
            if lor_count % 10000 == 0: print("processed", lor_count, "LORs")
            ## TODO attennuation and normalisation options
            if atn:
                lor = (0.0, pos1[0] * np.cos(pos1[1]), pos1[0] * np.sin(pos1[1]), pos1[2],
                            pos2[0] * np.cos(pos2[1]), pos2[0] * np.sin(pos2[1]), pos2[2])
                atn_corr = attenuation_correction(lor)
                binOut.write(struct.pack('<f', atn_corr))
            binOut.write(struct.pack('<I', i))
            binOut.write(struct.pack('<I', j + i+1))
            lor_count += 1
    binOut.close()
    return lor_count


def make_normalisation_header(outfile, geom_name, lor_count, atn):
    header = """Data filename: {OUTFILE}
Number of events: {NLORS}
Data mode: normalization
Data type: PET
Scanner name: {GEOM}
Attenuation correction flag: {ATN}
Normalization correction flag: 0"""
    with open(outfile[:-1] + 'h', 'w') as hdr_out:
        hdr_out.write(header.format(OUTFILE = outfile  ,
                                    NLORS   = lor_count,
                                    GEOM    = geom_name,
                                    ATN     = atn*1    ))


if __name__ == '__main__':
    args      = docopt(__doc__)

    geom_name = args['--geom']
    geom_arr  = get_geometry_histogram(geom_name, edges=False)
    ang_sep   = float(args['--ang']) * np.pi / 180.0
    atn       = args['--atn']
    outname   = args['OUTPUT']

    lor_count = generate_normalisation_lors(outname, geom_arr, ang_sep)
    make_normalisation_header(outname, geom_name, lor_count, atn)