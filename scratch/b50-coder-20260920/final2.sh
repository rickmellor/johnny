#!/usr/bin/bash
cd ~/repos/johnny/scratch/b50-coder-20260920
python3 runhard.py sycl-q2k
echo "HE sycl-q2k: $(./humaneval.sh sycl-q2k)"
echo FINALDONE
