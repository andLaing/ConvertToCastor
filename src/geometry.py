import os
import struct

from numba import njit, jit

import configparser as confp
import numpy        as np

from . utils import circular_coordinates


def get_geom_info(args):
    geom_lut = ''
    if args['--geom']:
        geom_base = args['--geom']
        # Check if we've been given a valid file name
        # Can be the base, the .hscan file or the .lut file.
        pos_geom  = (geom_base, geom_base+'.hscan', geom_base[:-5]+'lut')
        if not any(map(os.path.isfile, pos_geom)):
            raise RuntimeError('No valid geometry file found for {}.\
                                Provide configuration or existing file as name up to . or .hscan/lut'.format(geom_base))
        if '.hscan' in geom_base:
            geom_lut = geom_base[:-5] + 'lut'
        elif '.lut' in geom_base:
            geom_lut = geom_base
        else:
            geom_lut = geom_base + '.lut'
    else:
        print('Getting parameters to generate geometry')
        geom_conf = confp.ConfigParser()
        geom_conf.read(args['--conf'])

        geom_lut = generate_geometry(args['--conf'], geom_conf)

    #geom_arr = get_geometry_histogram(geom_lut)
    geom_params = get_geometry_parameterisation(geom_lut)

    return geom_lut, geom_params


def generate_geometry(conf_name, geom_conf):
    out_path = geom_conf['output'].get('fldr_path')
    out_base = geom_conf['output'].get('name_base')
    hdr_name = '/'.join((out_path, out_base + '.hscan'))
    lut_name = '/'.join((out_path, out_base + '.lut'  ))

    # Global dimensions
    inner_rad  = geom_conf['dimensions'].getfloat('inner_rad')
    det_length = geom_conf['dimensions'].getfloat('det_length')
    det_depth  = geom_conf['dimensions'].getfloat('det_thickness')

    # The number of layers etc
    nlayer = geom_conf['subdet'].getint('nlayer')
    lthick = [float(t) for t in geom_conf['subdet'].get('layer_depth').split(',')]
    cry_axial = geom_conf['subdet'].getfloat('crystal_axial')
    cry_trans = geom_conf['subdet'].getfloat('crystal_trans')

    ncryAx, ncryLay, cryAx, ang_diff = make_lutfile(lut_name, inner_rad, det_length,
                                                    lthick  , cry_axial, cry_trans )

    make_hscan(hdr_name, conf_name, geom_conf, ncryLay ,
               ncryAx  , lthick   , cryAx    , ang_diff)
    return lut_name


def make_lutfile(filename, radius, length, thicknesses, axial_dim, trans_dim):
    ## Adjust the dimensions so that everything fits.
    ncry_axial  = int(np.floor(length / axial_dim))
    cry_axial   = length / ncry_axial
    ncry_layer  = []
    adiff_layer = []
    radii_layer = []
    rad = radius
    for thick in thicknesses:
        rad      += thick / 2.0
        arc_theta = trans_dim / rad
        ncry      = int(np.floor(2 * np.pi / arc_theta))
        adiff_layer.append(2 * np.pi / ncry)
        ncry_layer .append(ncry)
        radii_layer.append(rad)
        rad      += thick / 2.0

    OrientZ = 0.0
    with open(filename, 'wb') as bin_out:
        for iring in range(ncry_axial):
            ring_pos = -(length / 2) + cry_axial * (iring + 0.5)
            for nc_lay, ad_lay, rad in zip(ncry_layer, adiff_layer, radii_layer):
                for icry in range(nc_lay):
                    ## Start theta 0 bin at angular diff / 2
                    ## to help look-up for data.
                    c_ang = np.cos(ad_lay / 2.0 + icry * ad_lay)
                    s_ang = np.sin(ad_lay / 2.0 + icry * ad_lay)
                    bin_out.write(struct.pack('<f', rad * c_ang))
                    bin_out.write(struct.pack('<f', rad * s_ang))
                    bin_out.write(struct.pack('<f', ring_pos   ))
                    bin_out.write(struct.pack('<f',       c_ang))
                    bin_out.write(struct.pack('<f',       s_ang))
                    bin_out.write(struct.pack('<f',     OrientZ))
    return ncry_axial, ncry_layer, cry_axial, adiff_layer


def make_hscan(filename  , conf_name  , gconf , ncry_layer,
               ncry_axial, thicknesses, ax_sep, ang_sep   ):
    header = """modality: PET
scanner: {SCANNER}
description: Geometry file generated using {CONFFILE}
number of elements: {TOTELEM}
number of layers: {NLAYER}
voxels number axial: {NVOXAXIAL}
voxels number transaxial: {NVOXTRANS}
field of view axial: {FOVAXIAL}
field of view transaxial: {FOVTRANS}
number of crystals in layer: {NCRYLAYER}
crystals size depth: {CRYTHICK}
#### These are variables to aid data conversion ####
#### angular separation: {ANGSEP}
#### axial separation: {AXSEP}
#### cryst. per ring per layer: {NCRYRL}
####################################################"""
    tot_elem  = sum(ncry_axial * ncry_layer)
    tot_perL  = [ncry_axial * ncry for ncry in ncry_layer]
    with open(filename, 'w') as hscan:
        hscan.write(header.format(SCANNER   = conf_name.split('/')[-1][:-5]      ,
                                  CONFFILE  = conf_name                          ,
                                  TOTELEM   = tot_elem                           ,
                                  NLAYER    = len(ncry_layer)                    ,
                                  NVOXAXIAL = gconf['image'].getint('nvox_axial'),
                                  NVOXTRANS = gconf['image'].getint('nvox_trans'),
                                  FOVTRANS  = gconf['image'].getint('fov_trans') ,
                                  FOVAXIAL  = gconf['image'].getint('fov_axial') ,
                                  NCRYLAYER = ', '.join(map(str, tot_perL   ))   ,
                                  CRYTHICK  = ', '.join(map(str, thicknesses))   ,
                                  ANGSEP    = ', '.join(map(str, ang_sep    ))   ,
                                  AXSEP     = ax_sep                             ,
                                  NCRYRL    = ', '.join(map(str, ncry_layer ))   ))


def get_geometry_histogram(lut_filename, edges=True):
    # get some info from the header.
    if edges:
        with open(lut_filename[:-3] + 'hscan') as hdrin:
            for line in hdrin:
                if 'crystals size depth' in line:
                    depths = list(map(float, line.split(':')[-1].split(',')))
                elif 'angular separation' in line:
                    angles = list(map(float, line.split(':')[-1].split(',')))
                elif 'axial separation' in line:
                    axlen  = float(line.split(':')[-1].split(',')[-1])
                elif 'cryst. per ring' in line:
                    ncryst = list(map(int  , line.split(':')[-1].split(',')))
    else:
        ## We just want the crystal positions.
        depths = [0]
        angles = [0]
        axlen  =  0
        ncryst = [1]
    # Get the crystal positions and calculate upper edges
    # in cylindrical coordinates.
    csum_cryst  = np.cumsum(ncryst)
    upper_edges = []
    with open(lut_filename, 'rb') as lutin:
        lut_data = lutin.read()
        for i, (xp, yp, zp, *_) in enumerate(struct.iter_unpack('<ffffff', lut_data)):
            ilayer = list(map(lambda x: (i % csum_cryst[-1]) // x, csum_cryst)).index(0)
            r, theta = circular_coordinates(xp, yp)
            upper_edges.append((r     + depths[ilayer] / 2.0,
                                theta + angles[ilayer] / 2.0,
                                zp    + axlen          / 2.0))
    return np.array(upper_edges)


def get_geometry_parameterisation(lut_filename):
    with open(lut_filename[:-3] + 'hscan') as hdrin:
        for line in hdrin:
            if 'crystals size depth' in line:
                depths = list(map(float, line.split(':')[-1].split(',')))
            elif 'angular separation' in line:
                angles = list(map(float, line.split(':')[-1].split(',')))
            elif 'axial separation' in line:
                axlen  = float(line.split(':')[-1].split(',')[-1])
            elif 'cryst. per ring' in line:
                ncryst = list(map(int  , line.split(':')[-1].split(',')))
    # Get the inner radius of the detector from the first crystal R.
    with open(lut_filename, 'rb') as lutin:
        lut_data       = lutin.read(24)
        xp, yp, zp, *_ = struct.unpack('<ffffff', lut_data)
        r, _       = circular_coordinates(xp, yp)
    return r - depths[0] / 2, zp - axlen / 2, depths, angles, axlen, ncryst



@njit
def in_bin(bin_edge, r, theta, z):
    return r < bin_edge[0] and theta < bin_edge[1] and z < bin_edge[2]

def get_crystal_index(geom_bins):
    """
        Look up in the bin limits for the crystals for the corresponding
        index. This is a bit slow for geometries with lots of crystals
        but allows for easy coding for general geometries. Maybe should
        change to using some parametrisation according to the crystals
        per layer etc.
    """
    @njit
    def get_cindex(r, theta, z):
        for i, pos in enumerate(geom_bins):
            if in_bin(pos, r, theta, z):
                return i
        return -1
    return get_cindex


def get_index_param(det_rad, det_zmin, depths, angSep, axSep, ncryst):
    ncry_ring   = sum(ncryst)
    csum_depths = np.cumsum([0] + depths)
    csum_ncryst = np.cumsum([0] + ncryst)
    r_max       = det_rad  + csum_depths[-1]
    ## Assume symetric about 0 (fine? should be improved)
    z_max       = -det_zmin
    def get_cindex(r, theta, z):
        if (r < det_rad) or (r > r_max) or (z < det_zmin) or (z > z_max): return -1
        ring  = int(np.floor((z - det_zmin) / axSep))
        layer = list(map(lambda x: r - det_rad >= x, csum_depths)).index(0) - 1
        rang  = int(np.floor(theta / angSep[layer]))
        return ring * ncry_ring + csum_ncryst[layer] + rang
    return get_cindex


def ray_func(lor):
    lor_np  = np.array(lor[1:4])
    dir_np  = np.array([lor[4] - lor[1], lor[5] - lor[2], lor[6] - lor[3]])
    dir_np /= np.linalg.norm(dir_np)
    def position(par):
        return lor_np + par * dir_np
    return dir_np, position


@jit
def get_path_through_phantom(lor):
    """
    Cylinder at centre of FOV
    """
    z_c = 186 / 2 #mm
    r_c = 216 / 2 #mm
    dir, ray = ray_func(lor)
    face1_t  = ( z_c - lor[3]) / (lor[6] - lor[3])
    face2_t  = (-z_c - lor[3]) / (lor[6] - lor[3])
    pos_f1   = ray(face1_t)
    pos_f2   = ray(face2_t)
    in_face1 = pos_f1[0]**2 + pos_f1[1]**2 < r_c**2
    in_face2 = pos_f2[0]**2 + pos_f2[1]**2 < r_c**2
    if in_face1 and in_face2:
        return np.linalg.norm(pos_f1 - pos_f2)
    
    determ, t1, t2 = cylinder_body_intersect(r_c, ray(0), dir)
    if determ < 0:
        #print("No intersection found.")
        return 0.0

    if in_face1:
        t_body = max(abs(t1 - face1_t), abs(t2 - face1_t))
        pos2   = ray(t_body)
        return np.linalg.norm(pos_f1 - pos2)
    if in_face2:
        t_body = max(abs(t1 - face2_t), abs(t2 - face2_t))
        pos2   = ray(t_body)
        return np.linalg.norm(pos_f2 - pos2)

    pos1 = ray(t1)
    pos2 = ray(t2)
    if abs(pos1[2]) > z_c or abs(pos2[2]) > z_c:
        #print("No intersection found")
        return 0.0
    return np.linalg.norm(pos1 - pos2)


@njit
def cylinder_body_intersect(r_cyl, ray0, ray_dir):
    """
    Parameters for intersection with the body of
    an axially infinte cylinder.
    """
    body_a = ray_dir[0]**2 + ray_dir[1]**2
    body_b = 2 * (ray_dir[0] * ray0[0] + ray_dir[1] * ray0[1])
    body_c = ray0[0]**2 + ray0[1]**2 - r_cyl**2
    determ = body_b**2 - 4 * body_a * body_c
    if determ < 0:
        return -1, None, None

    t1 = (-body_b + np.sqrt(determ)) / (2 * body_a)
    t2 = (-body_b - np.sqrt(determ)) / (2 * body_a)
    return determ, t1, t2


@jit
def get_path_through_steel(lor):
    """
    Assumes valid LOR so that the steel will always
    be traversed in the body of the cylinder.
    """
    dir, ray      = ray_func(lor)
    ray0          = ray(0)
    steel0R_inner = 325   #mm
    steel0R_outer = 326.5 #mm

    _, t1_inner, t2_inner = cylinder_body_intersect(steel0R_inner, ray0, dir)
    _, t1_outer, t2_outer = cylinder_body_intersect(steel0R_outer, ray0, dir)
    steel0_path = abs(t1_inner - t1_outer) + abs(t2_inner - t2_outer)

    steel1R_inner = 351.5 #mm
    steel1R_outer = 353   #mm

    _, t1_inner, t2_inner = cylinder_body_intersect(steel1R_inner, ray0, dir)
    _, t1_outer, t2_outer = cylinder_body_intersect(steel1R_outer, ray0, dir)
    steel1_path = abs(t1_inner - t1_outer) + abs(t2_inner - t2_outer)

    return steel0_path + steel1_path


@jit
def attenuation_correction(lor, steel=False):
    """
    Get the attenuation correction factor for a 
    given LOR. Assumes the attenuation for water
    and a cylinder of Jaszczak size.
    Needs to be generalised!!
    """
    atn_const = 0.0096 # mm^-1
    atn_steel = 0.0653 # mm^-1
    h2O_path  = get_path_through_phantom(lor)
    if steel and h2O_path > 0:
        steel_exp = atn_steel * get_path_through_steel(lor)
    else:
        steel_exp = 0.0
    return np.exp(atn_const * h2O_path + steel_exp)

#def define_roi(phi, theta, cry1_centre, )