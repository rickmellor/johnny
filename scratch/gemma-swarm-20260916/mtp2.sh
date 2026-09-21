#!/usr/bin/bash
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
until grep -q MTP1DONE logs/mtp1.out; do sleep 10; done; docker rm -f box-0 box-1 box-2 >/dev/null 2>&1
N=nightly-27a94d1ce4e3fc100c4732439ccec10f8246a804
./box4.sh MTP2 "tp2-mtp4-v0280-seqs32 v0.28.0 2 26 0.95 32 4 0 0" "tp2-mtp4-nightly-seqs32 $N 2 26 0.95 32 4 0 0" "tp2-mtp3-v0280-seqs32 v0.28.0 2 26 0.95 32 3 0 0"
echo MTP2DONE
