# PIMSimulator for LUT-Kernel

## Contents

  [1. Overview](#1-overview)  
  [2. Simulation Description](#2-simulation-description)  
  [3. How to use it](#3-how-to-use-it)  

## 1. Overview

### 1. PIMSimulator
[PIMSimulator](https://github.com/SAITPublic/PIMSimulator) is a cycle accurate model that Single Instruction, Multiple Data (SIMD) execution units
that uses the bank-level parallelism in PIM Block to boost performance that would have otherwise used
multiple times of bandwidth from simultaneous access of all bank.
The simulator include memory and have embedded within it a PIM block, which consist of programmable
command registers, general purpose register files and execution units.
The simulator can simulate HBM-PIM and other PIM systems that have the similar architecture. 

### 2. LUT-Kernel
[LUT-Kernel](https://arxiv.org/abs/2302.03213) uses the `index_matrix` and `LUT_table` as input, and generate `output_matrix` consisting of features.

The dimension of `LUT_table` is `(num_codebooks*num_centroids, out_features)`. `LUT_table[0:num_centroids][0:out_features]` means the first submatrix whose dimension is `(num_centroids, out_features)`, where it has `num_centroids` features from what a feature can be picked according to the `index_matrix`. 

The dimension of `index_matrix` is `(n, num_codebooks)`, where `n` means the number of features in the `output_matrix`, `num_codebooks` means the row stride of `LUT_table`. And each data of `index_matrix` is in `[0, num_centroids)`, which is the row index to the feature in the submatrix with `(num_centroids, out_features)` dimensions. Each row haves `num_codebooks` elems in `index_matrix`, which means the number of the features from different submatrixes divided by the row stride is `num_codebooks`. And finally, the picked features should be reduced into the `output_matrix`,  whose dimension is `(n, out_features)`.

According to HBM-PIM ISA limitation, HBM-PIM cannot directly extract features from `LUT_table`. To offload LUT-Kernel on HBM-PIM, `index_matrix` is extended to the bigger matrix, named `mask_matrix`, whose dimension is `(n, num_codebooks*num_centroids)`. The row vector whose length in `index_matrix` is `num_codebooks` is extended to the vector whose length is `num_codebooks*num_centroids`. The single element in `index_matrix` is extended into the `mask_vector` with `num_centroids` elements, where `mask_vector[i]` is set `1`, `i` is the data value in the `index_matrix` and left elements in `mask_vector` are all zero. After extension, LUT-Kernel is equal to `mask_matrix.matmul(LUT_table)`.

## 2. Simulation Description

PIMSimulator can simulate HBM-PIM behaviours including Write/Read memory, mode change and CRF program. There are more details about HBM-PIM in [HBM-PIM](https://ieeexplore.ieee.org/abstract/document/9499894) and [Function-In-Memory DRAM](https://ieeexplore.ieee.org/abstract/document/9365862).

PIMSimulator provides two mode tests:
- Functionality test: It reads data file and simulates kernel to generate correct results. Its ouput cycle contains the write banks from data files, kernel run simulation and read data from banks. 
- Performance test: It just simulates kernel behaviours.

I change the data generation files and test files, and edit new scripts to get LUT-Kernel performance on HBM-PIM fast for all kinds of inputs.

## 3. How to use it

### 3.1 Build first

#### 3.1.1 Prerequisites
* `Scons` tool for compiling PIMSimulator:
```bash
sudo apt install scons
```
* `gtest` for running test cases (https://github.com/google/googletest):
```bash
git clone https://github.com/google/googletest.git
cd googletest/googletest
cmake ..
make
sudo make install
```

**Notes:** You may meet the error that reminds you should add `-fPIE` option when you install googletest. You can edit `CMakeList.txt` to add corresponding flags:

#### 3.1.2 Installing
* To Install PIMSimulator:
```bash
# compile
scons
```

### 3.2 Get LUT-Kernel performance
The inputs for LUT-Kernel are `NUM_CODEBOOKS`, `NUM_TOKENS`, `NUM_CENTROIDS` and `FEATURE_LENS`.

If you want to get LUT-Kernel performance fast, you can type command like that:

```
python3 run_gemv_bench_test.py --num_codebooks 1 --num_tokens 2 --num_centroids 3 --feature_lens 4
```
It means the `NUM_CODEBOOKS=1`, `NUM_TOKENS=2`, `NUM_CENTROIDS=3` and `FEATURE_LENS=4` in LUT-Kernel.

In the end, the results will show in the terminal like:
```
  GEMV (PIM enabled)
  Weight data dimension : 4x3
  Input data dimension : 3x2
  Output data dimension : 4x2
> Test Results 
> Cycle : 1169
```
The cycle in `GEMV (PIM enabled)` means the LUT-Kernel performance in certain input.

At the end of terminal, the data related with energy shows like that:
```
memsReadNum: 10240
memsWriteNum: 2752
memsActivateNum: 3136
memsPrechargeNum: 3136
memsRefNum: 0
memsMacNum: 73728
memsAllIdleCycles: 104576
memsActiveCycles: 45056
background_energy: 0.039967(mJ)
act_energy: 0.012983(mJ)
read_energy: 0.041165(mJ)
write_energy: 0.014696(mJ)
ref_energy: 0.000000(mJ)
mac_energy: 1.492255(uJ)
total_energy: 0.110302(mJ)
power: 94.356283(J/s)
```

`total_energy` means the total energy of HBM-PIM on LUT-Kernel, and `power` means HBM-PIM power in LUT-Kernel process.

As well, if you want to get the LUT-Kernel functionality performance, you can type command like that:
```
python3 run_gemv_kernel_test.py --num_codebooks 1 --num_tokens 2 --num_centroids 3 --feature_lens 4
```
You can get the cycle in terminal finally, which consists write data to memory, kernel run and read data from memory.

### 3.3 When set LARGE input parameters

You should notice that if you set large input parameters, which will make PIMSimulator allocate dramatically huge space and simulation time is very long.

I recommand you can apply the some small inputs to drive simulator to get performance and use the results to model the LUT-Kernel performance in LARGE input parameters. In my tests, the simulator performance is almost same as the model performance.