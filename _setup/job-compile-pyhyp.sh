#!/bin/bash
#SBATCH --job-name=compyh
#SBATCH --time=0-08:00:00  # Job time limit Days-Hours:Minutes:Seconds
##SBATCH --exclusive  # Request entire nodes
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8      # Number of Cores per Task
##SBATCH --ntasks              # Number of MPI processes
##SBATCH --ntasks-per-node=32  # 
##SBATCH --constraint=ib # for infiniband
##BATCH --mem=32G  #  Memory
#SBATCH --mail-user=ebranlard@umass.edu
#SBATCH --mail-type END,FAIL # Send e-mail when job begins, ends or fails
#SBATCH --output=slurm-%x.log   # Output %j: job number, %x: jobname
echo "# Working directory $SLURM_SUBMIT_DIR"
echo "# Job name:         $SLURM_JOB_NAME"
echo "# Job ID:           $SLURM_JOBID"
echo "# Starting job on:  $(date)"

# --------------------- INPUT ----------------------------
EXAWIND_PARENT_DIR=/work/pi_ebranlard_umass_edu/
ENV_NAME=nalu-wind-shared
CREATE_ENV="true"

LIB_DIR=~/libs/

# ------------------- MODULES ----------------------------
echo "# Modules: "
module purge
#module load openmpi/5.0.3

echo "export MPI_INSTALL_DIR=~/libs/openmpi-5.0.7/opt" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_INSTALL_DIR/lib" >> ~/.bashrc
echo "export PATH=$MPI_INSTALL_DIR/bin:$PATH" >> ~/.bashrc
echo "" >>~/.bashrc
echo "export PETSC_ARCH=linux-debug" >> ~/.bashrc
echo "export PETSC_DIR=~/libs/petsc" >> ~/.bashrc
echo "" >>~/.bashrc
echo "export CGNS_HOME=~/libs/CGNS/opt/" >> ~/.bashrc
echo "export PATH=$PATH:$CGNS_HOME/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CGNS_HOME/lib" >> ~/.bashrc
echo "" >>~/.bashrc
echo "export PY_PATH_LOC_BIN=~/.local/bin" >> ~/.bashrc
echo "export PATH=$PATH:$PY_PATH_LOC_BIN" >> ~/.bashrc



echo "-------------------------- OPENMPI --------------------------"
echo "MPI_INSTALL_DIR : $MPI_INSTALL_DIR"
cd $LIB_DIR
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.gz
tar -xvaf openmpi-5.0.7.tar.gz 
cd openmpi-5.0.7
mkdir opt
./configure --prefix=$MPI_INSTALL_DIR
make all install
which mpicc
echo $MPI_INSTALL_DIR/bin/mpicc



echo "--------------------------- PETSC ---------------------------"
echo "PESC_DIR : $PETSC_DIR"
echo "PESC_ARCH: $PETSC_ARCH"
cd $LIB_DIR
git clone https://github.com/petsc/petsc 
cd petsc
git checkout v3.21.0
./configure --PETSC_ARCH=$PETSC_ARCH --with-scalar-type=real --with-debugging=1 --with-mpi-dir=$MPI_INSTALL_DIR --download-metis=yes --download-parmetis=yes --download-superlu_dist=yes --with-shared-libraries=yes --with-fortran-bindings=1 --with-cxx-dialect=C++11
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH test




echo "--------------------------- CGNS ----------------------------"
echo "CGNS_HOME: $CGNS_HOME"
cd $LIB_DIR
git clone https://github.com/CGNS/CGNS 
cd CGNS
git checkout v4.5.0
cmake -D CGNS_ENABLE_FORTRAN=ON -D CMAKE_INSTALL_PREFIX=$CGNS_HOME -D CGNS_ENABLE_64BIT=OFF -D CGNS_ENABLE_HDF5=OFF -D CGNS_BUILD_CGNSTOOLS=OFF -D CMAKE_C_FLAGS="-fPIC" -D CMAKE_Fortran_FLAGS="-fPIC" .
# cmake <options> -D CMAKE_C_COMPILER=/path/to/ccompiler -D CMAKE_Fortran_COMPILER=/path/to/fcompiler .
make install

# sudo apt-get install libxmu-dev libxi-dev
# sudo apt-get install freeglut3
# sudo apt-get install tk8.6-dev
# sudo apt-get install freeglut3-dev



echo "--------------------------- PYTHON --------------------------"
cd $LIB_DIR
source ~/pyenv/bin/activate
pip install numpy==1.26
# sudo apt-get install python3-dev gfortran valgrind cmake libblas-dev liblapack-dev build-essential swig
# sudo apt-get install python-is-python3
# sudo apt-get install f2py3
# sudo apt-get install python3-numpy
#pip install scipy==1.15


echo "-------------------------- PYSPLINE -------------------------"
cd $LIB_DIR
git clone https://github.com/mdolab/pyspline
cd pyspline/
cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk
make
#pip install .
pip install .


echo "-------------------------- PYGEO -------------------------"
cd $LIB_DIR
git clone https://github.com/mdolab/pygeo/
cd pygeo/
cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk
pip install .

#pip install .[testing]
#pip install .[testing]
#./tests/ref/get-ref-files.sh

echo "-------------------------- CGNS UTILITIES -----------------"
cd $LIB_DIR
git clone https://github.com/mdolab/cgnsutilities 
cd cgnsutilities
cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk
make
pip install .

echo "-------------------------- BASECLASSES  ------------------"
cd $LIB_DIR
git clone https://github.com/mdolab/baseclasses
cd baseclasses
pip install .



echo "-------------------------- PYHYP -------------------------"
cd $LIB_DIR
git clone https://github.com/mdolab/pyhyp/
cd pyhyp/
cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk
pip install .[testing]
testflo -v


#cd pyhyp/
#testflo -v

































































echo "# Ending job on:  $(date)"
