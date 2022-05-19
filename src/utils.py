import sys
import math

import numpy  as np
import tables as tb


def read_lors(filename, tbl_name, min_max_it=(0, sys.maxsize), min_eng=0):
    with tb.open_file(filename) as h5in:
        for i, lor in enumerate(h5in.root[tbl_name]):
            if i == min_max_it[1]: break
            if i >= min_max_it[0]:
                if min(*lor[-2:]) <= min_eng:
                    continue
                yield lor


def check_number_lors(filename, tbl_name):
    with tb.open_file(filename) as h5in:
        return h5in.root[tbl_name].shape[0]


def save_scattergram(filename, edges, values, duration):
    """
    Output scattergram to a file.
    """
    # TODO fix name!
    group_name = 'rzphi' if len(edges) == 3 else 'dtrzphi'
    with tb.open_file(filename, 'a') as h5out:
        grp = h5out.create_group(h5out.root, group_name)
        # Add the time of data as an attribute
        grp._v_attrs.duration = duration
        for i, edg in enumerate(edges):
            h5out.create_array(grp, 'bin_edges' + str(i), obj=edg, shape=edg.shape)
        h5out.create_array(grp, 'scatterPerSecond', obj=values, shape=values.shape)


def convert_to_lor_space(lor):
    """
    Convert into 4 dim LOR space (r, phi, (z1+z2)/2, pi)
    where:
        r   : Distance of closest approach to z axis.
        phi : Angle between x axis and r.
        pi  : Angle between transverse plane and LOR.
        returns a tuple including these values, dt and E1, E2 (temp for true scatter)
        dt, r, phi, z, pi, E1, E2
    """
    z_lor  = (lor[3] + lor[6]) / 2
    dx     =  lor[4] - lor[1]
    dy     =  lor[5] - lor[2]
    r_lor  = abs(dx * lor[2] - dy * lor[1]) / np.sqrt(dx * dx + dy * dy)
    phi    = np.arctan2(dx, dy)
    if phi < 0:
        # No need for 2pi as back-to-back gammas, -phi equivalent to pi-phi.
        phi = np.pi + phi
    r1     = np.sqrt(lor[1] * lor[1] + lor[2] * lor[2])
    r2     = np.sqrt(lor[4] * lor[4] + lor[5] * lor[5])
    trans1 = r1 * np.sin(np.arctan2(lor[2], r1) - phi)
    trans2 = r2 * np.sin(np.arctan2(lor[5], r2) - phi)
    theta  = np.arctan2(lor[6] - lor[3], trans2 - trans1)
    if theta < 0:
        # No need for 2pi as back-to-back gammas, -theta equivalent to pi-theta.
        theta = np.pi + theta
    return lor[0], r_lor, phi, z_lor, theta, *lor[-2:]


def make_scattergram(lors_space, dts=None):
    """
    Make a histogram for r, z and theta of a lor with
    the TOF included if requested.
    """
    # Time normalisation dummy for now
    time_dur = len(lors_space) * 0.001 #1 ms per event for tests.
    # True scatter at the moment.
    mask = np.amax(lors_space[:, -2:], axis=1) < 511.0
    cols = [1, 3, 4]
    bins = [np.linspace(   0,   385, 20        ),
            np.arange  (-500,   500, 50        ),
            np.arange  (   0, np.pi, np.pi / 10)]
    if dts:
        cols.insert(0, 0)
        bins.insert(0, [-20, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5, 20])
    H, edges = np.histogramdd(lors_space[np.ix_(mask, cols)], bins=bins)
    return edges, H, time_dur


def circular_coordinates(x, y):
    r     = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x) if y >= 0 else 2 * math.pi + math.atan2(y, x)
    return r, theta


def accepted_range(args):
    eng_ext = tuple(map(float, args['--eng'].split(','))) if args['--eng'] else (-math.inf, math.inf)
    q_ext   = tuple(map(float, args['--q'  ].split(','))) if args['--q'  ] else (-math.inf, math.inf)
    def lor_in_range(q1, q2, E1, E2):
        in_engrange = math.isnan(E1) or math.isnan(E2) or\
                      ((eng_ext[0] < E1 < eng_ext[1]) and (eng_ext[0] < E2 < eng_ext[1]))
        in_qrange   = math.isnan(q1) or math.isnan(q2) or\
                      ((  q_ext[0] < q1 <   q_ext[1]) and (  q_ext[0] < q2 <   q_ext[1]))
        return in_engrange and in_qrange
    return lor_in_range


def read_scattergram(filename_gen, tof=None):
    """
    Read in and normalise the scattergram and
    return a function to extract info.
    """
    # TODO fix name!
    grp_name  = 'rzphi'
    ndim      = 3
    bin_names = ['bin_edges'+str(i) for i in range(3)]
    time_dur  = 0.0
    hist_name = 'scatterPerSecond' # TODO fix name!! Not per second anymore
    if tof:
        grp_name  = 'dtrzphi'
        ndim     += 1
        bin_edges.append('bin_edges3')
    with tb.open_file(next(filename_gen)) as h5first:
        time_dur += h5first.root[grp_name]._v_attrs.duration
        bin_edges = [h5first.root['/'.join((grp_name, edges))].read() for edges in bin_names]
        scatters  = h5first.root['/'.join((grp_name, hist_name))].read()
    for fn in filename_gen:
        with tb.open_file(fn) as histIn:
            time_dur += histIn.root[grp_name]._v_attrs.duration
            scatters += histIn.root['/'.join((grp_name, hist_name))].read()
    # Normalise by the total duration
    scatters /= time_dur
    def get_scatter_value(lor_vec):
        indx = tuple(np.argmax(bins>x) - 1 for bins, x in zip(bin_edges, lor_vec))
        return scatters[indx]
    return get_scatter_value

