#!/bin/bash
D=~/repos/johnny/scratch/qwen38-flash-next-20260826
{
$D/lb.sh F-ple-on-gpu0 0,1,2,3 -fa 1 -ot "per_layer_token_embd=ROCm0" -ts 1,12,12,12
$D/lb.sh G-t4 0,1,2,3 -fa 1 -t 4
$D/lb.sh H-t24 0,1,2,3 -fa 1 -t 24
echo MATRIX2 DONE
} > $D/lb-matrix2.out 2>&1
