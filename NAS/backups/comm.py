import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import sys

from models import Cifar10RunConfig
from nas_manager import *
from models.super_nets.super_proxyless import SuperProxylessNASNets
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import Reducer
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP
    has_apex = False

from torchsummary import summary
from torch.profiler import profile, record_function, ProfilerActivity

def run(input_var, super_net, rank, size):
    """ Distributed function to be implemented later. """
    n_trials = 200
    reducer = Reducer(super_net) # no unused parameters
    
    super_net.init_arch_params(
        'normal', 1e-3,
    )
    super_net.unused_modules_back()
    criterion = nn.MSELoss().to(torch.device(rank))
    optimizer = torch.optim.SGD(super_net.weight_parameters(), 0.001, momentum=0.9,weight_decay = 0.001, nesterov=True)
    
    with torch.no_grad():
        x = super_net(input_var)
    
    output_var = torch.zeros(x.shape, device = torch.device(rank)).detach()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    totals = []
    res = []
    for i in range(n_trials): 
        torch.cuda.synchronize()
        torch.distributed.barrier()
        super_net.reset_binary_gates()  # random sample binary gates
        super_net.unused_modules_off()  # remove unused module for speedup
        
        x = input_var.detach()
        # Super Net
        x = super_net(x)
        loss = criterion(x, output_var) 
        super_net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
        loss.backward()
        torch.cuda.synchronize()

        start.record()
        reducer.reduce()
        
        end.record()
        torch.cuda.synchronize()
        if i > n_trials/10:
            res.append(start.elapsed_time(end))

        super_net.unused_modules_back()

        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))
        #print('[my_rank',rank, ']\n',prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
        
        #if rank == 0:
        #    print('Total: %.4f'%(sum(totals)/len(totals)))
        #    print('Comm.: %.4f'%(sum(res)/len(res)))
    
    if rank == 0:
        #print('[Total: %.4f]'%(sum(totals)/len(totals)))
        print('[Comm.: %.4f]'%(sum(res)/len(res)))
    return output_var
    

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=size)
    conv_candidates = [
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
    ]

    data_shape = [256, 3,32,32]
    input_var = torch.zeros(data_shape, device = torch.device(rank))
    for block_idx in range(6):
        super_net = SuperProxylessNASNets(
                width_stages=[24,32,64,96,160,320], n_cell_stages=[4,4,4,4,4,1], stride_stages=[1,2,2,1,2,1],
                conv_candidates=conv_candidates, n_classes=10,
            )
        
        super_net.unused_stages_off(block_idx, block_idx)
        super_net.to(rank)
        #if rank == 0:
        #    summary(super_net, tuple(list(input_var.shape)[1:]), batch_size = 64 )
        MixedEdge.MODE = None
        if has_apex:
            pass
            #net = convert_syncbn_model(super_net)
            #net = DDP(net, delay_allreduce=True) # no unused parameters
        else:
            print("Using torch DistributedDataParallel, install apex")
            raise NotImplementedError
        output_var = fn(input_var, super_net, rank, size)
        input_var = output_var




if __name__ == "__main__":
    size = 4
    processes = []
    
    manual_seed = 0
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    np.random.seed(manual_seed)

    # (3,224,224)
    for rank in range(size):
        p = mp.Process(target=init_process, args=( rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


