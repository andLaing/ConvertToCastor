import functools
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


def save_scattergram(filename, edges, scat_vals, all_vals, duration):
    """
    Output scattergram to a file.
    """
    group_name = 'rzth' if len(edges) == 3 else 'dtrzth'
    with tb.open_file(filename, 'a') as h5out:
        grp = h5out.create_group(h5out.root, group_name)
        # Add the time of data as an attribute
        grp._v_attrs.duration = duration
        for i, edg in enumerate(edges):
            h5out.create_array(grp, 'bin_edges' + str(i), obj=edg, shape=edg.shape)
        h5out.create_array(grp, 'scatter', obj=scat_vals, shape=scat_vals.shape)
        h5out.create_array(grp, 'allLOR' , obj=all_vals , shape=all_vals .shape)


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
    phi    = np.arctan2(dx, -dy)
    trans1 = lor[2] * np.cos(phi) - lor[1] * np.sin(phi)
    trans2 = lor[5] * np.cos(phi) - lor[4] * np.sin(phi)
    theta  = np.arctan2(lor[6] - lor[3], trans2 - trans1)
    if phi < 0:
        # No need for 2pi as back-to-back gammas, -phi equivalent to pi-phi.
        phi = np.pi + phi
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
    mask = np.amin(lors_space[:, -2:], axis=1) < 511.0
    cols = [1, 3, 4]
    #bins = [np.linspace(   0,   385, 20),
    #        np.linspace(-500,   500, 20),
    #        np.linspace(   0, np.pi, 20)]
    bins = [np.concatenate((np.linspace(0.0, 250, 68), np.linspace(270,   370,  4))),
            np.concatenate((np.linspace(-500, -250, 5), np.linspace(-240, 250, 49), np.linspace(300, 500, 4))),
            np.concatenate((np.linspace(0.0, 1.0, 20), np.linspace(2.0, np.pi, 21)))]
    if dts:
        cols.insert(0, 0)
        #bins.insert(0, [-20, -2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2, 20])
        bins.insert(0, np.concatenate([[-20, -1.5, -1.25], np.linspace(-1.2, 1.2, 20), [1.25, 1.5, 20]]))
    HScat, edges = np.histogramdd(lors_space[np.ix_(mask, cols)], bins = bins)
    HAll , _     = np.histogramdd(lors_space[:          , cols ], bins = bins)
    return edges, HScat, HAll, time_dur


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
    grp_name  = 'rzth'
    ndim      = 3
    bin_names = ['bin_edges'+str(i) for i in range(3)]
    time_dur  = 0.0
    scat_name = 'scatter'
    all_name  = 'allLOR'
    if tof:
        grp_name  = 'dtrzth'
        ndim     += 1
        bin_names.append('bin_edges3')
    with tb.open_file(next(filename_gen)) as h5first:
        time_dur += h5first.root[grp_name]._v_attrs.duration
        bin_edges = [h5first.root['/'.join((grp_name, edges))].read() for edges in bin_names]
        scatters  = h5first.root['/'.join((grp_name, scat_name))].read()
        all_lor   = h5first.root['/'.join((grp_name,  all_name))].read()
    for fn in filename_gen:
        with tb.open_file(fn) as histIn:
            time_dur += histIn.root[grp_name]._v_attrs.duration
            scatters += histIn.root['/'.join((grp_name, scat_name))].read()
            all_lor  += histIn.root['/'.join((grp_name,  all_name))].read()
    # Normalise by the total duration
    scatters /= time_dur
    # Average per lor bin
    scatters = np.divide(scatters, all_lor, out=np.zeros_like(scatters), where=all_lor!=0)
    # Normalise to bin sizes allowing for non-even bins
    bin_norm = functools.reduce(lambda a, b: a[..., np.newaxis] * np.diff(b),
                                bin_edges[1:], np.diff(bin_edges[0]))
    scatters /= bin_norm    
    def get_scatter_value(lor_vec):
        indx = tuple(np.argmax(bins>x) - 1 for bins, x in zip(bin_edges, lor_vec))
        return scatters[indx]
    return get_scatter_value

