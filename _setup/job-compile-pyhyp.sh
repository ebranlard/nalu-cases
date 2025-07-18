#!/bin/bash
#SBATCH --job-name=compyh
#SBATCH --time=0-01:00:00  # Job time limit Days-Hours:Minutes:Seconds
##SBATCH --exclusive  # Request entire nodes
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4      # Number of Cores per Task
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

# ------------------ DOCUMENTATION -----------------------
# This script install all the dependencies for pyhyp (mesher). 
# There are only two inputs, LIB_DIR and ENV_FILE. 
# 
# It can be run multiple times, it won't reinstalled what's already installed.
# It's inspired from the steps listed here:
#    https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/installInstructions/install3rdPartyPackages.html#installthirdpartypackages
# When installing packages, it's important to use the dedicated version they support. This script does that.

# --------------------- INPUTS ----------------------------
#LIB_DIR=/work/pi_ebranlard_umass_edu/libs/
LIB_DIR=~/libs
ENV_FILE=~/.pyhyp_env



# Python environment
PYENV_DIR="$LIB_DIR/pyenv"
mkdir -p $LIB_DIR


# --- Helper functions
check_file() {
    local file="$1"
    if [ -f "$file" ]; then
        echo "[ OK ] Found  $file"
    else
        echo "[FAIL] Missing: $file"
        exit "$exit_code"
    fi
}




echo "---------------------- REQUIREMENTS -------------------------"
#echo "# Modules: "
#module purge
#module load openmpi/5.0.3

# List of required commands
required_commands=("gfortran" "gcc" "cmake" "valgrind" "swig")
missing=0
for cmd in "${required_commands[@]}"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "[FAIL] $cmd not found."
        missing=1
    else
        echo "[ OK ] $cmd found."
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "[FAIL] One or more required commands are missing."
    echo "       Please install the following dependencies before running this script:"
    echo ""
    echo "sudo apt-get install python3-dev gfortran valgrind cmake libblas-dev liblapack-dev build-essential swig python-is-python3 python3-venv"
# sudo apt-get install python3-dev gfortran valgrind cmake libblas-dev liblapack-dev build-essential swig
# sudo apt-get install python-is-python3
# sudo apt-get install f2py3
# sudo apt-get install python3-numpy
    echo ""
    exit 1
else
    echo "[ OK ] Commands already present."
fi





if [ -f $ENV_FILE ]; then
    echo "---------------------LOADING ENV FILE -----------------------"
    echo "[ OK ] ENV FILE FOUND"
else 
    echo "------------------ SETUP BASH AND ENV FILE-------------------"
    echo "" >>$ENV_FILE
    echo "export MPI_INSTALL_DIR=$LIB_DIR/openmpi-5.0.7/opt" >>$ENV_FILE
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$MPI_INSTALL_DIR/lib" >> $ENV_FILE
    echo "export PATH=\$MPI_INSTALL_DIR/bin:\$PATH" >> $ENV_FILE
    echo "" >>$ENV_FILE
    echo "export PETSC_ARCH=linux-debug" >>$ENV_FILE
    echo "export PETSC_DIR=$LIB_DIR/petsc" >> $ENV_FILE
    echo "" >>$ENV_FILE
    echo "export CGNS_HOME=$LIB_DIR/CGNS/opt" >> $ENV_FILE
    echo "export PATH=\$PATH:\$CGNS_HOME/bin" >> $ENV_FILE
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CGNS_HOME/lib" >> $ENV_FILE
    echo "" >>$ENV_FILE
    echo "export PY_PATH_LOC_BIN=~/.local/bin" >> $ENV_FILE
    echo "export PATH=\$PATH:\$PY_PATH_LOC_BIN" >> $ENV_FILE

    echo "" >>$ENV_FILE
    echo "echo \"MPI_INSTALL_DIR : \$MPI_INSTALL_DIR\" " >> $ENV_FILE
    echo "echo \"CGNS_HOME       : \$CGNS_HOME\"              " >> $ENV_FILE
    echo "echo \"PESC_DIR        : \$PETSC_DIR\"              " >> $ENV_FILE
    echo "echo \"PESC_ARCH       : \$PETSC_ARCH\"             " >> $ENV_FILE

    echo "" >>$ENV_FILE
    echo "if [ -f $ENV_FILE ]; then  " >> ~/.bashrc
    echo "   source $ENV_FILE " >> ~/.bashrc
    echo "fi" >> ~/.bashrc
fi

echo "ENV_FILE        : $ENV_FILE"
echo "LIB_DIR         : $LIB_DIR"
source $ENV_FILE

required_env_vars=("MPI_INSTALL_DIR" "CGNS_HOME" "PETSC_DIR" "PETSC_ARCH")
# Check if any are unset
for var in "${required_env_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "[FAIL] $var environment variable not set."
        exit 1
    fi
done
echo "[ OK ] ENV FILE and ENV VAR LOADED"

#echo "MPI_INSTALL_DIR : $MPI_INSTALL_DIR"
#echo "CGNS_HOME:        $CGNS_HOME"     
#echo "PESC_DIR :        $PETSC_DIR"    
#echo "PESC_ARCH:        $PETSC_ARCH"  


echo "-------------------------- OPENMPI --------------------------"
MPICC_PATH="$MPI_INSTALL_DIR/bin/mpicc"
echo "MPI_INSTALL_DIR : $MPI_INSTALL_DIR"
echo "MPI_CC_PATH     : $MPICC_PATH"
cd $LIB_DIR
if [ -d $MPI_INSTALL_DIR ]; then
   echo "[ OK ] MPI_INSTALL_DIR EXISTS"
else
   wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.gz
   tar -xvaf openmpi-5.0.7.tar.gz 
   rm openmpi-5.0.7.tar.gz 
   cd $LIB_DIR/openmpi-5.0.7
   mkdir -p opt
fi;
if [ ! -x "$MPICC_PATH" ]; then
    echo "[INFO] mpicc not found at: $MPICC_PATH"
    cd $LIB_DIR/openmpi-5.0.7
    echo ">>> CONFIGURE OPENMPI"
    ./configure --prefix=$MPI_INSTALL_DIR
    echo ">>> MAKE OPENMPI"
    make -j 12 all install 
    which mpicc
    echo $MPI_INSTALL_DIR/bin/mpicc
fi
if [ -x "$MPICC_PATH" ]; then
    echo "[ OK ] MPI installed correctly at: $MPICC_PATH"
else
    echo "[FAIL] MPI not installed correctly"
    exit -1
fi
 
 
 
echo "--------------------------- PETSC ---------------------------"
PETSC_LIB=$PETSC_DIR/$PETSC_ARCH/lib/libpetsc.so 
echo "PETSC_DIR : $PETSC_DIR"
echo "PETSC_ARCH: $PETSC_ARCH"
echo "PETSC_LIB : $PETSC_LIB"
cd $LIB_DIR
if [ -d $PETSC_DIR ]; then
    echo "[ OK ] PETSC_DIR EXISTS"
else
    git clone https://github.com/petsc/petsc 
    cd petsc
    git checkout v3.21.0
fi
if [ ! -f "$PETSC_LIB" ]; then
    cd $LIB_DIR/petsc
    echo ">>> CONFIGURE PETSC"
    git describe
    ./configure --PETSC_ARCH=$PETSC_ARCH --with-scalar-type=real --with-debugging=1 --with-mpi-dir=$MPI_INSTALL_DIR --download-metis=yes --download-parmetis=yes --download-superlu_dist=yes --with-shared-libraries=yes --with-fortran-bindings=1 --with-cxx-dialect=C++11
    echo ">>> MAKE PETSC"
    make -j 12 PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all
    #make -j 12 PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH test
fi
check_file "$PETSC_LIB"




echo "--------------------------- CGNS ----------------------------"
CGNS_LIB=$CGNS_HOME/lib/libcgns.a
echo "CGNS_HOME: $CGNS_HOME"
echo "CGNS_LIB : $CGNS_LIB"
cd $LIB_DIR
if [ -d $CGNS_HOME ]; then
    echo "[ OK ] CGNS_HOME EXISTS"
else
    git clone https://github.com/CGNS/CGNS 
    cd CGNS
    mkdir opt
    git checkout v4.5.0
fi
if [ ! -f /home/ebranlard/libs/CGNS/opt/lib/libcgns.a ]; then
    cd $LIB_DIR/CGNS
    git describe
    cmake -D CGNS_ENABLE_FORTRAN=ON -D CMAKE_INSTALL_PREFIX=$CGNS_HOME -D CGNS_ENABLE_64BIT=OFF -D CGNS_ENABLE_HDF5=OFF -D CGNS_BUILD_CGNSTOOLS=OFF -D CMAKE_C_FLAGS="-fPIC" -D CMAKE_Fortran_FLAGS="-fPIC" .
    make -j 12 install
fi
check_file "$CGNS_LIB"

# # sudo apt-get install libxmu-dev libxi-dev
# # sudo apt-get install freeglut3
# # sudo apt-get install tk8.6-dev
# # sudo apt-get install freeglut3-dev
# 
# # 
# 
echo "------------------------ PYTHON ENV --------------------------"
# Check if environment exists
if [ ! -d "$PYENV_DIR" ]; then
    echo "[INFO] Python environment not found, creating it..."
    python -m venv "$PYENV_DIR"
    if [ $? -ne 0 ]; then
        echo "[FAIL] Failed to create Python environment at $PYENV_DIR"
        rm -rf "$PYENV_DIR"
        exit 1
    fi
fi
#  Activate environment
source "$PYENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo "[FAIL] Failed to activate Python environment"
    exit 1
fi
echo "[ OK ] Python environment activated at $PYENV_DIR"


echo "------------------------ PYTHON --------------------------"
pip install numpy==1.26
pip install scipy==1.15

echo "-------------------------- PYSPLINE -------------------------"
PYSPLINE_LIB=$LIB_DIR/pyspline/lib/libspline.a
cd $LIB_DIR
if [ -d pyspline ]; then
    echo "[ OK ] pysline EXISTS"
else
    git clone https://github.com/mdolab/pyspline
fi
if [ ! -f $PYSPLINE_LIB ]; then
    cd $LIB_DIR/pyspline/
    cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk
    make -j 12
    pip install .
fi
check_file "$PYSPLINE_LIB"


echo "-------------------------- PYGEO -------------------------"
cd $LIB_DIR
if [ -d pygeo ]; then
    echo "[ OK ] pygeo EXISTS"
else
    git clone https://github.com/mdolab/pygeo/
    cd $LIB_DIR/pygeo/
    pip install .
fi


echo "-------------------------- CGNS UTILITIES -----------------"
CGNSU_LIB=$LIB_DIR/cgnsutilities/build/lib/cgnsutilities/libcgns_utils.so
cd $LIB_DIR
if [ -d cgnsutilities ]; then
    echo "[ OK ] cgnsutilities EXISTS"
else
    git clone https://github.com/mdolab/cgnsutilities 
fi
if [ ! -f $CGNSU_LIB ]; then
    cd $LIB_DIR/cgnsutilities
    cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk
    make -j 12
    pip install .
fi
check_file "$CGNSU_LIB"

echo "-------------------------- BASECLASSES  ------------------"
cd $LIB_DIR
if [ -d baseclasses ]; then
    echo "[ OK ] baseclasses EXISTS"
else
    git clone https://github.com/mdolab/baseclasses
    cd $LIB_DIR/baseclasses
    pip install .
fi



echo "-------------------------- PYHYP -------------------------"
PYHYP_LIB=$LIB_DIR/pyhyp/lib/libhyp.a
cd $LIB_DIR
if [ -d pyhyp ]; then
    echo "[ OK ] pyhyp EXISTS"
else
    git clone https://github.com/mdolab/pyhyp/
fi
if [ ! -f $PYHYP_LIB ]; then
    cd $LIB_DIR/pyhyp/
    cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk
    sed -i '/^FF90_GEN_FLAGS *= / s/$/ -fallow-argument-mismatch/'  config/config.mk
    make -j 12
    pip install .
fi
check_file "$PYHYP_LIB"


echo "---------------------- PYHYP TEST -------------------------"
python $LIB_DIR/pyhyp/examples/naca0012/naca0012_euler.py
check_file $LIB_DIR/pyhyp/examples/naca0012/naca0012_euler.cgns




# 
# 
## testflo -v
## #cd pyhyp/
## #testflo -v

echo "--------------------------  END  -------------------------"
echo "# Ending job on:  $(date)"
