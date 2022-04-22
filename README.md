# ConvertToCastor
Some python scripts to aid conversion of HDF5 tables to CASToR binaries

# Requirements

Uses packages:
HDF5
numba
numpy
pytables
docopt
struct
configparser

# make_castor.py

Script to convert an HDF5 input to a castor binary and header output.

Can either use an existing geometry `.lut` file or be passed a configuration
to make the relevant geometry file too.

python make_castor -h for usage.

# make_normalisation.py

Script to make a CASToR normalisation file from an existing geometry `.lut` file.
