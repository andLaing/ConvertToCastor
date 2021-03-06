#!/usr/bin/env python3

"""Convert LOR file to CASToR file format

Usage: make_castor.py (--geom GEOMFILE | --conf CONFFILE) [--out OUTFILE]
                      [--nlor NLOR] [--tbl INTABLE] [--tof TOFR] [--atn]
                      [--eng ENGRNG] [--q CHRGRNG] [--scat SCATF] [--norm NORMF] INPUT...

Arguments:
    INPUT hdf5 table files with LORs

Options:
    --geom=GEOMFILE File with the geometry to be used for indexing the LORS.
                    If not provided uses name provided in conf to build and
                    save geometry.
    --conf=CONFFILE Configuration file with detector dimensions etc used to
                    generate geometry file before conversion. Required if
                    geom nor used.
    --out=OUTFILE   Name with path for outfile. If not provided uses
                    INPUT_GEOMFILE.cdh/cdf
    --nlor=NLOR     Maximum number of LORs to process, default all found.
    --tbl=INTABLE   Name of the table in the input where the LORs are
                    stored, defaul 'reco_info/lors' [default: reco_info/lors]
    --eng=ENGRNG    Max,Min true energy of interactions to allow in output.
                    Default all LORs.
    --q=CHRGRNG     Max,Min reconstructed charge of interactions to allow in output.
                    Default all LORs.
    --tof=TOFR      The TOF resolution in ps for the reconstruction.
                    If not used, no TOF info in the data files.
    --atn           Include attenuation prediction in output.
    --scat=SCATF    Correct for scatter using the histogrammed info predictions
                    in the provided file(s).
    --norm=NORMF    If LOR normalisation factors to be included filename for
                    lookup.
"""

import math

from docopt import docopt
from glob   import iglob

from src.castor_data import make_data_header
from src.castor_data import make_data_binary
from src.geometry    import get_geom_info
from src.geometry    import get_index_param
from src.utils       import accepted_range
from src.utils       import read_scattergram


if __name__ == '__main__':
    args = docopt(__doc__)

    in_files = args['INPUT']
    
    geom_lut, geom_pars = get_geom_info(args)
    #indices             = get_crystal_index(geom_arr)
    indices             = get_index_param(*geom_pars)

    out_file  =     args['--out' ]  if args['--out']  else geom_lut[:-3] + 'cdf'
    max_lors  = int(args['--nlor']) if args['--nlor'] else math.inf
    tbl_name  =     args['--tbl' ]
    tof_res   =     args['--tof' ]
    atn       =     args['--atn' ]
    scat      =     args['--scat']
    norm      =     args['--norm']
    eng_range = accepted_range(args)
    scat_pred = read_scattergram(iglob(scat + '*'), tof_res) if scat else None
    # TODO can we have multiple files or are we obliged to have one
    # per run?
    lor_count, tof_rng = make_data_binary(in_files, out_file ,
                                          tbl_name, max_lors ,
                                          indices , tof_res  ,
                                          atn     , scat_pred,
                                          norm    , eng_range)
    
    # Now the header.
    make_data_header(out_file, geom_lut.split('/')[-1][:-4],
                     lor_count, tof_res, tof_rng, atn, scat, norm)
                
