#!/usrbin/env bash

# Make a conda environment for the convertor assuming conda installed
# in home folder.
CONDA_SH=$HOME/miniconda/etc/profile.d/conda.sh
source $CONDA_SH

CONDA_ENV_NAME=cConvert
YML_FILENAME=environment-${CONDA_ENV_NAME}.yml

echo creating ${YML_FILENAME}

cat <<EOF > ${YML_FILENAME}
- python       = 3.8
- docopt       = 0.6.2
- hdf5         = 1.12.1
- numba        = 0.55.1
- numpy        = 1.21.2
- pytables     = 3.7.0
- configparser = 5.0.2
EOF

conda env create -f ${YML_FILENAME}
conda activate ${CONDA_ENV_NAME}


