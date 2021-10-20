# ASpT (ppopp19_AE)

This repository contains materials for artifact evaluation of the paper "Adaptive Sparse Tiling for Sparse Matrix Multiplication".  It contains:

1) Source code for ASpT (Adaptive Sparse Tiling)
2) Scripts for installation of ASpT and other compared implementations: MKL, CSB, TACO, cuSPARSE, Merge-SpMM, and BIDMach
3) The entire data set for BIDMach 
4) The experimental results at ppopp19_ae_result.xlsx

## Datasets


- Run `./download.sh` to download all datasets. The minimum required space is 200 GB
- BIDMach uses custom data format. The entire dataset in BIDMach format is included in the repository itself 

Note: Downloading and running full dataset may take more than 24 hours. Hence for convenience, we have provided `./download_small.sh` which downloads only 6 datasets

## Installation and Benchmarking


### SpMM & SDDMM KNL:

- Run `cd nlibs; make; cd ..` 
- Run  `./compile_KNL.sh` to compile all implementations
- Run `./run_KNL.sh` to run all implementations (results are available on KNL_result folder)

#### Notes:
- `INTEL_PATH` (line 1) and `MKL_FLAGS` (line 2) in `Makefile.in` should be set properly before installation
- Make sure that the compiler and library path for Intel compiler and MKL is set properly (modify `line 1 in run_KNL.sh` )

#### Software requirements:

    1. Intel ICC 18.0.3 (with MKL) 

#### Hardware requirements:

    1. Intel Xeon Phi (AVX 512, 68 cores, 272 threads)
    2. Clustering mode: 'All-to-All'
    3. Memory mode: 'cache-mode'


### SpMM GPU:

- Run  `./compile_GPU_SpMM.sh` to compile all implementations. 
- Run `./run_GPU_SpMM.sh` to run all implementations (results are available on GPU_SpMM_result folder)

#### Notes: 

- For some datasets, Merge-SpMM throws errors and hence some results may not be available 
- The performance of cuSPARSE in `CUDA 9.1.85` is higher than `CUDA 8.0.44`. Hence for fairness, we used `CUDA 9.1.85` for comparing our framework with cuSPARSE. However, for SDDMM, since BIDMach only supports `CUDA 8.*`, we used `CUDA 8.0.44`


#### Software requirements:

    1. NVCC 9.1.85
    2. GNU 4.8.5 or 4.9.3
    3. Cmake 3.11.4
    4. Boost 1.58

#### Hardware requirements:

    1. Nvidia Pascal-P100 GPU  
    2. Compute Capability 6.0
    3. Global Memory >= 16 GB

### SDDMM GPU:

- Run `./compile_GPU_SDDMM.sh` to compile ASpT implementations 
- Installation instruction for BIDMach is provided below 
- Run './run_GPU_SDDMM.sh' to run implementations (results are availble on GPU_SDDMM_result folder).

#### Note: 

- The performance of cuSPARSE in `CUDA 9.1.85` is higher than `CUDA 8.0.44`. Hence for fairness, we used `CUDA 9.1.85` for comparing our framework with cuSPARSE. However, for SDDMM, since BIDMach only supports `CUDA 8.*`, we used `CUDA 8.0.44`

#### Software requirements:

    1. NVCC 8.0.44 (BIDMach only supports CUDA 8.0)
    2. GNU 4.8.5 (for installiation), GNU 6.1.0 (for execution)
    3. Apache Maven 3.6.0
    4. JDK 8

#### Hardware requirements:

    1. Nvidia Pascal-P100 GPU 
    2. Compute Capability 6.0
    3. Global Memory >= 16 GB

#### BIDMach installiaton:
    0. mkdir build; cd build;
    1. wget https://downloads.apache.org/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz
    2. tar xf apache-maven-3.6.3-bin.tar.gz
    3. export M2_HOME=`readlink -f apache-maven-3.6.3`
    4. export M2=$M2_HOME/bin
    5. export PATH=$M2:$PATH
    5.1. After doing this, mvn -version should properly print the version.     Then go to this site : http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html - and download jdk-8u191-linux-x64.tar.gz (182.87 MB) and do the following.
    6. tar -xvf <name of downloaded file>
    7. export JAVA_HOME= <path to unzipped jdk directory>
    8. export PATH="$JAVA_HOME/bin:$PATH"
    8.1. After doing this "javac -version" should print javac 1.8X.
    9. Running './compile_GPU_SDDMM.sh'
