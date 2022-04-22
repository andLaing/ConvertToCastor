import struct

# with open('test_geom/H20body-steel_LXe30mm_1layer_6x6crystals_minE434_h2oAtn_100files.cdf', 'rb') as normIn:
with open('test_geom/normalisation_h20body_h20atn_Xe30mm_1layer_6x6crystals.cdf', 'rb') as normIn:
    norm_data = normIn.read()
    counter = 0
    i2s = []
    for atn, i1, i2 in struct.iter_unpack('<fII', norm_data):
        # if atn > 1:
        #     print(i1, i2, atn, t)
        counter += 1
    print("Total occurrences of atn > 1:", counter)