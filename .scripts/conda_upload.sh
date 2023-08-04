#!/bin/bash

PKG_NAME=httomolib
USER=DLS
OS=linux-64
CONDA_TOKEN=$(cat $HOME/.secrets/my_secret.json)

mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld

#for python_ver in 3.9 3.10; do   
#    for numpy_ver in 1.21 1.22 1.23 1.24; do
#        export VERSION=`date +%Y.%m`"_py"$python_ver"_np"$numpy_ver
#        conda build . --numpy $numpy_ver --python $python_ver
#    done
# done

$CONDA/bin/conda install conda-build
$CONDA/bin/conda install -c anaconda anaconda-client

$CONDA/bin/conda build . -c conda-forge -c httomo

# upload packages to conda
find $CONDA_BLD_PATH/$OS -name *.tar.bz2 | while read file
do
    echo $file
    $CONDA/bin/anaconda -v --show-traceback --token $CONDA_TOKEN upload --label $LABEL $file --force
done
