
# Method with exawind manager:
#      https://nrel.github.io/HPC/Documentation/Applications/exawind/
# Traditional method:  https://exawind.github.io/nalu-wind/source/user/build_spack.html

## Method with exawind manager:

salloc --time=01:00:00 --account=bar --partition=shared --nodes=1 --ntasks-per-node=52

# Intel
module load PrgEnv-intel
module load cray-mpich/8.1.28
module load cray-libsci/23.12.5
module load cray-python

# clone ExaWind-manager
cd /scratch/${USER}
git clone --recursive https://github.com/Exawind/exawind-manager.git
cd exawind-manager

# Activate exawind-manager
export EXAWIND_MANAGER=`pwd`
source ${EXAWIND_MANAGER}/start.sh && spack-start

# Create Spack environment and change the software versions if needed
mkdir environments
cd environments
# spack manager create-env --name naluwind-cpu --spec 'nalu-wind+hypre+netcdf %oneapi'
spack manager create-env --name naluwind-cpu --spec 'nalu-wind+hypre %oneapi'

# Activate the environment
spack env activate -d ${EXAWIND_MANAGER}/environments/naluwind-cpu

# concretize specs and dependencies
spack concretize -f

# Build software
spack -d install
