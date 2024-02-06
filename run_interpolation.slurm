#!/bin/bash

 

#SBATCH -p compute
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -e job-%j.err
#SBATCH -o job-%j.out
#SBATCH -J latent_space_interpolation
#SBATCH --hint=nomultithread
#SBATCH --mem=120GB
#SBATCH --mail-user=laura.balasso@optimad.it
#SBATCH --mail-type=END,FAIL
 

#module load
module purge
module load Compilers/mpi/openmpi-4.1.2 DevTools/python/3.6 DevLibs/boost/boost_1.78.0_complete LinearAlgebra/blas_lapack/blas_lapack_3.10.0 DevLibs/vtk/vtk-9.1.0_openMPI4.1.2
module load DevTools/toolworks/totalview_2021

 

source /home/laura/Programs/miniconda3/etc/profile.d/conda.sh
export PATH="home/laura/Programs/miniconda3/bin:$PATH"

conda activate deepsdf_env

 

echo 'loaded modules'
echo 'launch py script'
python latent_space_interpolation -e experiments/drivaer/2_var_no_wheel/07_run_full_samples
echo 'Done!'

 

conda deactivate
