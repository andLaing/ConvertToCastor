import sys
import math
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
    group_name = 'rzphi' if len(edges) == 3 else 'dtrzphi'
    with tb.open_file(filename, 'a') as h5out:
        grp = h5out.create_group(h5out.root, group_name)
        # Add the time of data as an attribute
        grp._v_attrs.duration = duration
        for i, edg in enumerate(edges):
            h5out.create_array(grp, 'bin_edges' + str(i), obj=edg, shape=edg.shape)
        h5out.create_array(grp, 'scatterPerSecond', obj=values, shape=values.shape)


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