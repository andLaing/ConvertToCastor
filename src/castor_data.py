import struct

from . geometry import attenuation_correction
from . geometry import interaction_norm
from . utils    import circular_coordinates
from . utils    import convert_to_lor_space
# from . utils    import normalisation_function
from . utils    import read_lors


def make_data_header(data_name, geom_name, nlors, tof_res, tof_rng, atn, scat, norm):
    if tof_res:
        tof_flag  = 1
        tof_fwhm  = tof_res
        ## defined as dtmax - dtmin
    else:
        tof_flag  = 0
        tof_fwhm  = 0
    scat_flag = 1 if scat else 0
    norm_flag = 1 if norm else 0
    data_short = data_name.split('/')[-1]
    header = """Scanner name: {SCANNAME}
Data filename: {DATAFILE}
Number of events: {NLOR}
Data mode: list-mode
Data type: PET
Start time (s): 0
Duration (s): {TOTTIME}
TOF information flag: {TOF}
TOF resolution (ps): {TOFFWHM}
List TOF measurement range (ps): {TOFRANGE}
Attenuation correction flag: {ATN}
Normalization correction flag: {NORM}
Scatter correction flag: {SCAT}
Random correction flag: 0"""
    with open(data_name[:-3] + 'cdh', 'w') as hdr_out:
        hdr_out.write(header.format(SCANNAME = geom_name    ,
                                    DATAFILE = data_short   ,
                                    NLOR     = nlors        ,
                                    TOTTIME  = nlors * 0.001,
                                    TOF      = tof_flag     ,
                                    TOFFWHM  = tof_fwhm     ,
                                    TOFRANGE = tof_rng      ,
                                    ATN      = atn*1        ,
                                    SCAT     = scat_flag    ,
                                    NORM     = norm_flag    ))


def make_data_binary(in_files, out_file ,
                     tbl_name, max_lors ,
                     indices , tof_res  ,
                     atn     , scat     ,
                     norm    , eng_range):
    lor_count = 0
    dt_min    = 0.0
    dt_max    = 0.0
    # if norm:
    #     norm_lookup = normalisation_function(norm)
    with open(out_file, 'wb') as data_out:
        for fn in in_files:
            print('Processing file ', fn)
            # TODO scatter.
            # TODO do something sensible with the timestamp.
            for lor in read_lors(fn, tbl_name):
                if not eng_range(*lor[7:]): continue
                t          = lor_count * 1 ## Fake timestamp each 1 ms
                r1, theta1 = circular_coordinates(lor[1], lor[2])
                z1         = lor[3]
                r2, theta2 = circular_coordinates(lor[4], lor[5])
                z2         = lor[6]

                cryst1 = indices(r1, theta1, z1)
                cryst2 = indices(r2, theta2, z2)
                if cryst1 < 0 or cryst2 < 0:
                    print(cryst1, "with position: ", (r1, theta1, z1))
                    print(cryst2, "with position: ", (r2, theta2, z2))
                    #raise KeyError('No index found for interaction.')
                    continue

                data_out.write(struct.pack('<I', t     ))
                if atn:
                    # atn_corr = attenuation_correction(lor, steel=True)
                    atn_corr = attenuation_correction(lor, steel=False)
                    data_out.write(struct.pack('<f', atn_corr))
                if scat:
                    (dt_lor, r_lor ,  _,
                     z_lor , th_lor, *_) = convert_to_lor_space(lor)
                    # This could clearly be done better.
                    lvec = [r_lor, z_lor, th_lor]
                    # if tof_res:
                    #     lvec.insert(0, dt_lor)
                    scat_corr = scat(lvec)
                    data_out.write(struct.pack('<f', scat_corr))
                if norm:
                    # lor_space = convert_to_lor_space(lor)
                    # norm_fact = norm_lookup(lor_space[1:-2])
                    norm_fact = interaction_norm(lor)
                    data_out.write(struct.pack('<f', norm_fact))
                # For TOF will need to invert dt
                # since dt = t2 - t1 (2022 standard)
                # since CASToR expects t1 - t2.
                # HDF5 has units of ns.
                # TODO use units to do conversion
                if tof_res:
                    dt = -lor[0] * 1000
                    if dt < dt_min:
                        dt_min = dt
                    elif dt > dt_max:
                        dt_max = dt
                    data_out.write(struct.pack('<f', dt))
                data_out.write(struct.pack('<I', cryst1))
                data_out.write(struct.pack('<I', cryst2))
                lor_count += 1
                if lor_count >= max_lors: break
                elif lor_count % 10000 == 0: print('Processed', lor_count, 'LORs')
    return lor_count, dt_max - dt_min