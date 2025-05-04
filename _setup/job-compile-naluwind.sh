#!/bin/bash
#SBATCH --job-name=CNW-shared
#SBATCH --time=0-08:00:00  # Job time limit Days-Hours:Minutes:Seconds
##SBATCH --exclusive  # Request entire nodes
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32     # Number of Cores per Task
##SBATCH --ntasks              # Number of MPI processes
##SBATCH --ntasks-per-node=32  # 
##SBATCH --constraint=ib # for infiniband
##BATCH --mem=32G  #  Memory
#SBATCH --mail-user=ebranlard@umass.edu
#SBATCH --mail-type END,FAIL # Send e-mail when job begins, ends or fails
#SBATCH --output=slurm-%x.log   # Output %j: job number, %x: jobname
## -q long
####SBATCH -G 1  # Number of GPUs
####SBATCH -p gpu  # Partition
####SBATCH --time=0-36
####SBATCH --account=bar
echo "# Working directory $SLURM_SUBMIT_DIR"
echo "# Job name:         $SLURM_JOB_NAME"
echo "# Job ID:           $SLURM_JOBID"
echo "# Starting job on:  $(date)"

# --------------------- INPUT ----------------------------
EXAWIND_PARENT_DIR=/work/pi_ebranlard_umass_edu/
#ENV_NAME=nalu-wind-nomod
# ENV_NAME=nalu-wind-openmpi
# ENV_NAME=nalu-wind-oneapi
ENV_NAME=nalu-wind-shared
CREATE_ENV="true"


# ------------------- MODULES ----------------------------
echo "# Modules: "
module purge
module load intel-oneapi-compilers/2024.1.0
#module load mpich/4.2.1
module load openmpi/5.0.3
#module load python/3.12.3
module list


# ------------- SETUP EXAWIND MANAGER -------------------
export EXAWIND_MANAGER=${EXAWIND_PARENT_DIR}/exawind-manager
echo "# EXAWIND_MANAGER: ${EXAWIND_MANAGER}"

if [ ! -d ${EXAWIND_PARENT_DIR} ]; then
    echo "# >>> The exawind parent directory does not exist"
    exit 1
fi
# --- Clone ExaWind-manager
if [ ! -d ${EXAWIND_MANAGER} ]; then
    echo "# >>> Cloning exawind-manager"
    cd ${EXAWIND_PARENT_DIR} || exit 1
    git clone --recursive https://github.com/Exawind/exawind-manager.git || exit 1
else
    echo "# >>> Exawind-manager present"
fi
# --- Creating environments directory
if [ ! -d ${EXAWIND_MANAGER}/environments ]; then
    echo "# >>> Creating environment directory"
    cd ${EXAWIND_MANAGER} || exit 1
    mkdir environments
else
    echo "# >>> Environments directory exists"
fi
# --- Activate Exawind
echo "# >>> Activating spack from exawind-manager"
source "${EXAWIND_MANAGER}/start.sh" && spack-start

# ------------- SETUP EXAWIND ENVIRONMENT ---------------
# --- Create Spack environment and change the software versions if needed
if [[ "$CREATE_ENV" == "true" ]]; then
    echo "# >>> Creating environment: ${ENV_NAME}"
    cd ${EXAWIND_MANAGER}/environments || exit 1
    spack manager create-env --name $ENV_NAME --spec 'nalu-wind+hypre+tioga+shared %oneapi' || exit 1
#     spack manager create-env --name $ENV_NAME --spec 'nalu-wind+hypre+tioga+trilinos-solvers %oneapi' || exit 1
else
    echo "# >>> Not creating dedicated environment"
fi
# --------------------------------------------------------

# ------------ ACTIVATE & INSTALL ENVIRONMENT -----------
echo "# >>> Activating environment  : ${ENV_NAME}"
spack env activate -d "${EXAWIND_MANAGER}/environments/${ENV_NAME}"  || exit 1

# --- Check dependencies 
echo "# >>> Spack concretize"
spack concretize -f || exit 1

echo "# >>> Spack install:"
spack install

echo "# Ending job on:  $(date)"
