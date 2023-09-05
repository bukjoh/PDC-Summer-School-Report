#!/bin/bash

export OMP_NUM_THREADS=$1
export OMP_PLACES=threads

srun -n1 energy_storms_omp 6000000 test_files/test_07_a1M_p5k_w1 test_files/test_07_a1M_p5k_w2 test_files/test_07_a1M_p5k_w3 test_files/test_07_a1M_p5k_w4
