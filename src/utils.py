import math
import tables as tb


def read_lors(filename, tbl_name):
    with tb.open_file(filename) as h5in:
        for lor in h5in.root[tbl_name]:
            yield lor


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