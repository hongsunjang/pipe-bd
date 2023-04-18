import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import sys

from torch.profiler import profile, record_function, ProfilerActivity


def run(feature_size, rank, size):
    """ Distributed function to be implemented later. """
    torch.cuda.set_device(rank)
    #tensor1 = torch.zeros((1), device= torch.device(rank))
    tensor1 = torch.zeros(feature_size, device= torch.device(rank))
    #print(tensor1.element_size())
    #print(tensor1.element_size() * tensor1.nelement())     

    res = [] 
    n_sample = 200
    

    for i in range(n_sample):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(torch.device(rank))
        start.record()

        if rank == 0:
            dist.send(tensor1, dst= 1)
        else:
            dist.recv(tensor1, src= 0)

        end.record()
        torch.cuda.synchronize(torch.device(rank))
        if i>= n_sample/ 10:
            res.append(start.elapsed_time(end))
        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    print('[rank %d] elapsed time: %.4f'%(rank, sum(res)/len(res)))
    

def init_process(feature_size, rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(feature_size, rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    feature_cand = [(1,), (256, 24, 56, 56), (256, 32, 28, 28), (256, 64, 14, 14), (256, 96, 14, 14), (256, 160, 7, 7)]
    for feature_size in feature_cand:
        for rank in range(size):
            p = mp.Process(target=init_process, args=(feature_size, rank, size, run))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


    
