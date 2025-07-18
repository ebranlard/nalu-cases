#!/bin/bash
#SBATCH --job-name=C
#SBATCH --time=0-08:00:00  # Job time limit Days-Hours:Minutes:Seconds
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32     # Number of Cores per Task
#SBATCH --mail-user=ebranlard@umass.edu
#SBATCH --mail-type END,FAIL # Send e-mail when job begins, ends or fails
#SBATCH --output=slurm-%x.log   # Output %j: job number, %x: jobname
####SBATCH --account=bar
echo "# Working directory $SLURM_SUBMIT_DIR"
echo "# Job name:         $SLURM_JOB_NAME"
echo "# Job ID:           $SLURM_JOBID"
echo "# Starting job on:  $(date)"

# --------------------- USER FLAGS ----------------------------
CLUSTER="home"      # unity, kestrel, or home
CONFIG="nalu-wind-shared"  # nalu-wind-shared, amr-wind, exawind-cpu, nalu-wind-gpu, amrwind-openfast-cpu, etc.

# --------------------- INPUTS BASED ON CLUSTER ---------------
if [[ "$CLUSTER" == "home" ]]; then
    EXAWIND_PARENT_DIR=/home/${USER}/
    ONEAPI=
elif [[ "$CLUSTER" == "kestrel" ]]; then
    EXAWIND_PARENT_DIR=/scratch/${USER}/
    ONEAPI=%oneapi
else
    EXAWIND_PARENT_DIR=/work/pi_ebranlard_umass_edu/
    ONEAPI=%oneapi
fi

# --------------------- SPECIFICATIONS -------------------------

case "$CONFIG" in
    nalu-wind-shared)
        SPACK_SPECS="nalu-wind+hypre+tioga+shared $ONEAPI"
        MODULES_KESTREL=
        MODULES_UNITY=intel-oneapi-compilers/2024.1.0  openmpi/5.0.3
        ;;
    nalu-wind-trilinos)
        SPACK_SPECS="nalu-wind+hypre+tioga+trilinos-solvers $ONEAPI"
        MODULES_KESTREL=
        MODULES_UNITY=intel-oneapi-compilers/2024.1.0  openmpi/5.0.3
        ;;
    nalu-wind-gpu)
        SPACK_SPECS="nalu-wind+hypre+tioga+shared+cuda+gpu-aware-mpi cuda_arch=90 %gcc"
        MODULES_KESTREL=PrgEnv-gnu cray-mpich/8.1.28 cray-libsci/23.12.5 cuda cray-python
        MODULES_UNITY=
        ;;
    amr-wind)
        SPACK_SPECS="amr-wind+hypre+netcdf $ONEAPI"
        MODULES_KESTREL= PrgEnv-intel cray-mpich/8.1.28 cray-libsci/23.12.5 cray-python
        MODULES_UNITY=
        ;;
    amrwind-openfast-cpu)
        SPACK_SPECS="amr-wind+hypre+netcdf+openfast ^openfast@develop+openmp+rosco $ONEAPI"
        MODULES_KESTREL= PrgEnv-intel cray-mpich/8.1.28 cray-libsci/23.12.5 cray-python
        MODULES_UNITY=
        ;;
    exawind-cpu)
        SPACK_SPECS="exawind@master~amr_wind_gpu~cuda~gpu-aware-mpi~nalu_wind_gpu ^amr-wind@main~cuda~gpu-aware-mpi+hypre+mpi+netcdf+shared ^nalu-wind@master~cuda~fftw~gpu-aware-mpi+hypre+shared ^tioga@develop $ONEAPI"
        MODULES_KESTREL= PrgEnv-intel cray-mpich/8.1.28 cray-libsci/23.12.5 cray-python
        MODULES_UNITY=
        ;;
    *)
        echo "# Unknown CONFIG: $CONFIG"
        exit 1
        ;;
esac

# -------------------- ECHO CONFIG -----------------------
export EXAWIND_MANAGER=${EXAWIND_PARENT_DIR}/exawind-manager
echo "# CONFIG         : ${CONFIG}"
echo "# CLUSTER        : ${CLUSTER}"
echo "# EXAWIND_MANAGER: ${EXAWIND_MANAGER}"
echo "# SPACK_SPECS    : ${SPACK_SPECS}"


# ------------------- MODULES ----------------------------
if [[ "$CLUSTER" == "unity" || "$CLUSTER" == "kestrel" ]]; then
    echo "# Modules: "
    module purge
    if [[ "$CLUSTER" == "kestrel" ]]; then
        module load $MODULES_KESTREL
    elif [[ "$CLUSTER" == "unity" ]]; then
        module load $MODULES_UNITY
    fi
    module list
fi

# ------------- SETUP EXAWIND MANAGER -------------------
if [ ! -d ${EXAWIND_PARENT_DIR} ]; then
    echo "# >>> The exawind parent directory does not exist"
    exit 1
fi
# --- Clone ExaWind-manager
if [ ! -d ${EXAWIND_MANAGER} ]; then
    echo "# >>> Cloning exawind-manager"
    cd ${EXAWIND_PARENT_DIR} || exit 1
    git clone --recursive https://github.com/Exawind/exawind-manager.git || exit 1
fi
# --- Creating environments directory
echo "# >>> Ensuring environment directory exists"
mkdir -p "${EXAWIND_MANAGER}/environments"

# --- Activate Exawind spack
echo "# >>> Activating spack from exawind-manager"
source "${EXAWIND_MANAGER}/start.sh" && spack-start

# ------------- SETUP EXAWIND ENVIRONMENT ---------------
echo "# >>> Creating spack environment: ${CONFIG}"
rm -rf ${EXAWIND_MANAGER}/environments/${CONFIG}
cd ${EXAWIND_MANAGER}/environments || exit 1
spack manager create-env --name $CONFIG --spec "$SPACK_SPECS" || exit 1
# --------------------------------------------------------

# ------------ ACTIVATE & INSTALL ENVIRONMENT -----------
echo "# >>> Activating environment  : ${CONFIG}"
spack env activate -d "${EXAWIND_MANAGER}/environments/${CONFIG}"  || exit 1

echo "# >>> Spack concretize"
spack concretize -f || exit 1

echo "# >>> Spack install:"
spack install

echo "# Ending job on:  $(date)"
