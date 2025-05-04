
# ------------- SETUP EXAWIND MANAGER -------------------
cd /work/pi_ebranlard_umass_edu/
git clone --recursive https://github.com/Exawind/exawind-manager.git || exit 1
export EXAWIND_MANAGER=/work/pi_ebranlard_umass_edu/exawind-manager
cd ${EXAWIND_MANAGER}
mkdir environments
# ------- LOAD SPACK AND CREATE ENV WITH SPECS -----------
source "${EXAWIND_MANAGER}/start.sh" && spack-start
spack manager create-env --name nalu-wind-shared --spec 'nalu-wind+hypre+tioga+shared %oneapi' 
# -------- LOAD SPACK ENV AND INSTALL --------------------
spack env activate nalu-wind-shared 
spack concretize -f || exit 1
spack install

# ---- RUN NALU WIND ON EXAMPLE (NEED ~200 CPU, 200Gb)-----
spack load nalu-wind
cd ~/test_large
mpiexec -n 64 naluX -i input.yaml
