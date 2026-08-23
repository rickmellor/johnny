import os, time, torch, torch.distributed as dist
def main():
    dist.init_process_group("nccl"); r=dist.get_rank(); torch.cuda.set_device(r)
    for nbytes in (64*1024, 1024*1024, 16*1024*1024):   # decode-sized → prefill-sized
        x=torch.ones(nbytes//2, dtype=torch.bfloat16, device="cuda")
        for _ in range(10): dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier(); t0=time.time(); n=50
        for _ in range(n): dist.all_reduce(x)
        torch.cuda.synchronize(); dt=(time.time()-t0)/n
        if r==0: print(f"all_reduce {nbytes//1024:>6} KiB: {dt*1e6:8.0f} us  ({nbytes/dt/1e9:6.2f} GB/s)", flush=True)
    dist.destroy_process_group()
if __name__=="__main__": main()
