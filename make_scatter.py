import sys

import numpy as np

from src.utils import check_number_lors
from src.utils import convert_to_lor_space
from src.utils import make_scattergram
from src.utils import read_lors
from src.utils import save_scattergram


def get_scatter(filename, outfile, dt):
    """
    Histogram scatter using MC info
    """
    min_eng = 434.0 #hardwire for first tests.
    # Let's chunk the input so we don't end up with loads
    # of lors in memory. brute force version.
    max_per_file = 5000000
    nchunk       = check_number_lors(filename, 'reco_info/lors') // max_per_file + 1
    for i in range(nchunk):
        evt_range = (i * max_per_file, (i + 1) * max_per_file)
        lors = np.array(tuple(map(convert_to_lor_space             ,
                                  read_lors(filename              ,
                                            'reco_info/lors'      ,
                                            min_max_it = evt_range,
                                            min_eng    = min_eng  ))))
        edges, scatter_rate, duration = make_scattergram(lors, dt)
        outname = outfile[:-3] + f'_{i}.h5'
        save_scattergram(outname, edges, scatter_rate, duration)


if __name__ == '__main__':
    filename = sys.argv[1]
    outfile  = sys.argv[2]
    dt       = sys.argv[3] if len(sys.argv) > 3 else None
    get_scatter(filename, outfile, dt)