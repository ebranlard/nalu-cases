
# Compiling:
See :
    https://exawind.github.io/nalu-wind/source/user/build_spack.html


## Installing spack
cd ${HOME} && git clone https://github.com/spack/spack.git

## Add this to your bashrc to have access to spack
export SPACK_ROOT=${HOME}/spack
source ${SPACK_ROOT}/share/spack/setup-env.sh


## install
spack install nalu-wind+trilinos-solvers+tioga


## Load a spacfic nalu-wind (because I installed two...)
spack load nalu-wind@2.1.0/oy


nalyX -i ffa_w3_211.yaml
