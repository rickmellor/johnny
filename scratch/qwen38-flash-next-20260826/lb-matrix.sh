#!/bin/bash
D=~/repos/johnny/scratch/qwen38-flash-next-20260826
{
$D/lb.sh A-4gpu-fa1 0,1,2,3 -fa 1
$D/lb.sh B-4gpu-fa0 0,1,2,3 -fa 0
$D/lb.sh C-3gpu-fa1 0,1,2 -fa 1
$D/lb.sh D-4gpu-smrow 0,1,2,3 -fa 1 -sm row
GGML_CUDA_DISABLE_GRAPHS=1 $D/lb.sh E-4gpu-nographs 0,1,2,3 -fa 1
echo MATRIX DONE
} > $D/lb-matrix.out 2>&1
