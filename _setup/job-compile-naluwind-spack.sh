#!/bin/bash
#SBATCH --job-name=CN
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
    SPACK_PARENT_DIR=/home/${USER}/
elif [[ "$CLUSTER" == "kestrel" ]]; then
    SPACK_PARENT_DIR=/scratch/${USER}/
else
    SPACK_PARENT_DIR=/work/pi_ebranlard_umass_edu/
fi

# --------------------- SPECIFICATIONS -------------------------
case "$CONFIG" in
    nalu-wind-shared)
        SPACK_SPECS="nalu-wind+hypre+tioga+shared"
        ;;
    nalu-wind-trilinos)
        SPACK_SPECS="nalu-wind+hypre+tioga+trilinos-solvers"
        ;;
    *)
        echo "# Unknown CONFIG: $CONFIG"
        exit 1
        ;;
esac

# -------------------- ECHO CONFIG -----------------------
export SPACK_ROOT=${SPACK_PARENT_DIR}/spack
export BUILD_ROOT=${SPACK_PARENT_DIR}/build-test
echo "# CONFIG         : ${CONFIG}"
echo "# CLUSTER        : ${CLUSTER}"
echo "# SPACK_ROOT     : ${SPACK_ROOT}"
echo "# SPACK_SPECS    : ${SPACK_SPECS}"

# ------------- SETUP SPACK MANAGER -------------------
if [ ! -d ${SPACK_PARENT_DIR} ]; then
    echo "# >>> The spack parent directory does not exist"
    exit 1
fi
# --- Clone SPACK
if [ ! -d ${SPACK_ROOT} ]; then
    echo "# >>> Cloning manager"
    cd ${SPACK_PARENT_DIR}  || exit 1
    git clone https://github.com/spack/spack.git || exit 1
fi
# --- Clone SPACK config for exawind
if [ ! -d ${BUILD_ROOT} ]; then
    echo "# >>> Cloning exawind config"
    cd ${SPACK_PARENT_DIR} 
    git clone https://github.com/exawind/build-test.git
fi

echo "# >>> Activating spack"
source ${SPACK_ROOT}/share/spack/setup-env.sh
cd ${BUILD_ROOT}/configs && ./setup-spack.sh

echo "# >>> Install spack"
#spack info nalu-wind
spack compiler find
spack compilers
spack install ${SPACK_SPECS}

