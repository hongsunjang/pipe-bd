# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import argparse
import datetime
import time
from models import ImagenetRunConfig
from models import Cifar10RunConfig

from nas_manager import *
from models.super_nets.super_proxyless import SuperProxylessNASNets
from build_mobilenet import MobileProxylessNASNets

from utils import *

# timm
from timm.models import create_model

from torchsummary import summary

import random

#import matplotlib.pyplot as plt

# ref values
ref_values = {
    'flops': {
        '0.35': 59 * 1e6,
        '0.50': 97 * 1e6,
        '0.75': 209 * 1e6,
        '1.00': 300 * 1e6,
        '1.30': 509 * 1e6,
        '1.40': 582 * 1e6,
    },
    # ms
    'mobile': {
        '1.00': 80,
    },
    'cpu': {},
    'gpu8': {},
}

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--debug', help='freeze the weight parameters', action='store_true')
parser.add_argument('--manual_seed', default=1, type=int)

""" run config """
parser.add_argument('--n_epochs', type=int, default=120)
parser.add_argument('--init_lr', type=float, default=0.025) 
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
# lr_schedule_param

parser.add_argument('--dataset', type=str, default='imagenet', choices=['cifar10','imagenet'])
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--valid_size', type=int, default=50000)

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str, default=None, choices=[None, 'bn', 'bn#bias'])

parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=5)

parser.add_argument('--n_worker', type=int, default=32)
parser.add_argument('--resize_scale', type=float, default=0.08)
parser.add_argument('--distort_color', type=str, default='normal', choices=['normal', 'strong', 'None'])

""" net config """
parser.add_argument('--width_stages', type=str, default='24,32,64,96,160,320')
parser.add_argument('--n_cell_stages', type=str, default='2,4,4,4,4,1')
parser.add_argument('--stride_stages', type=str, default='2,2,2,1,2,1')

parser.add_argument('--width_mult', type=float, default=1.0)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0)

# architecture search config
""" arch search algo and warmup """
parser.add_argument('--arch_algo', type=str, default='grad', choices=['grad', 'rl'])
parser.add_argument('--warmup_epochs', type=int, default=40)

""" shared hyper-parameters """
parser.add_argument('--arch_init_type', type=str, default='normal', choices=['normal', 'uniform'])
parser.add_argument('--arch_init_ratio', type=float, default=1e-3)
parser.add_argument('--arch_opt_type', type=str, default='adam', choices=['adam'])
#parser.add_argument('--arch_lr', type=float, default=1e-3) #arch_lr
parser.add_argument('--arch_lr', type=float, default=1e-3) #arch_lr
parser.add_argument('--arch_adam_beta1', type=float, default=0)  # arch_opt_param
parser.add_argument('--arch_adam_beta2', type=float, default=0.999)  # arch_opt_param
parser.add_argument('--arch_adam_eps', type=float, default=1e-8)  # arch_opt_param
parser.add_argument('--arch_weight_decay', type=float, default=0)
parser.add_argument('--target_hardware', type=str, default=None, choices=['mobile', 'cpu', 'gpu8', 'flops', None])

""" Grad hyper-parameters """
parser.add_argument('--grad_update_arch_param_every', type=int, default=5)
parser.add_argument('--grad_update_steps', type=int, default=1)
parser.add_argument('--grad_binary_mode', type=str, default='full_v2', choices=['full_v2', 'full', 'two'])
parser.add_argument('--grad_data_batch', type=int, default=None)
parser.add_argument('--grad_reg_loss_type', type=str, default='mul#log', choices=['add#linear', 'mul#log'])
parser.add_argument('--grad_reg_loss_lambda', type=float, default=1e-1)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_alpha', type=float, default=1)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_beta', type=float, default=3.6)  # grad_reg_loss_params

""" RL hyper-parameters """
parser.add_argument('--rl_batch_size', type=int, default=10)
parser.add_argument('--rl_update_per_epoch', action='store_true')
parser.add_argument('--rl_update_steps_per_epoch', type=int, default=300)
parser.add_argument('--rl_baseline_decay_weight', type=float, default=0.99)
parser.add_argument('--rl_tradeoff_ratio', type=float, default=0.1)

parser.add_argument('--train_method', type =str, required= True, choices=['dp', 'dna', 'mp', 'ts'])
parser.add_argument('--start_stage', type = int, default=0)
parser.add_argument('--scheme', type = str, default = 'ahd', choices=['ckpt','tr','sks','ahd'])
parser.add_argument('--dynamic', action='store_true')


def profile(args, arch_search_config = None, run_config = None, rank_to_in_feature_shape = None, stage_info = None, batch_to_t_profs = None ):
    my_rank = int(torch.distributed.get_rank())
    world_size = int(torch.distributed.get_world_size())
    
    print("Profile start at rank", my_rank)

    b_choices = [ int( args.train_batch_size/ r ) for r in range(1, world_size+1)]
    batch_size  = b_choices[my_rank]
    
    if args.dataset == 'cifar10':
        n_classes = 10
        data_shape = [batch_size] + list((3,32,32))
        build_teacher = build_cifar10_teacher
    elif args.dataset == 'imagenet':
        n_classes = 1000
        data_shape = [batch_size] + list((3,224,224))
        build_teacher = build_imagenet_teacher

    n_blocks = 6
    n_trials = 10 #* 6 * 6 # the maximum number of choice in one block
    s_profs = []

    if batch_to_t_profs is not None:
        t_profs = batch_to_t_profs[my_rank] 
    else:
        t_profs = []
        input_var = torch.zeros(data_shape, device = torch.device(my_rank))
        for block_idx in range(n_blocks):
            res = []
            teacher_net = build_teacher(args, run_config)

            teacher_net.unused_stages_off(block_idx, block_idx)
            teacher_net.to(torch.device(my_rank)) 

            for i in range(n_trials): 
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                with torch.no_grad():
                    start.record()
                    # Super Net
                    x = teacher_net(input_var)
                    end.record()
                    torch.cuda.synchronize()
                    if i > (n_trials / 10): 
                        elapsed_time = start.elapsed_time(end)
                        res.append(elapsed_time) 
            input_var = torch.zeros(x.shape, device = torch.device(my_rank)).detach()
            t_profs.append(sum(res)/len(res))
        teacher_net = None 
        torch.cuda.empty_cache()

    input_var = torch.zeros(tuple(data_shape), device = torch.device(my_rank))
    for block_idx in range(n_blocks):
        res = []
        try:
            if batch_to_t_profs is not None:
                super_net = load_full_student(args, stage_info, arch_search_config, run_config, rank_to_in_feature_shape)
            else:
                super_net = SuperProxylessNASNets(
                    width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
                    conv_candidates=args.conv_candidates, n_classes=n_classes, width_mult=args.width_mult,
                    bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
                )
            super_net.unused_stages_off(block_idx, block_idx)
            torch.cuda.empty_cache()
            super_net.to(torch.device(my_rank)) 
            MixedEdge.MODE = None
            if batch_to_t_profs is None:
                super_net.init_arch_params(
                    args.arch_init_type, args.arch_init_ratio,
                )
            super_net.unused_modules_back()
            criterion = nn.MSELoss().to(torch.device(my_rank))
        
            optimizer = torch.optim.SGD(super_net.weight_parameters(), 0.001, momentum=0.9,weight_decay = 0.001, nesterov=True)
            
            with torch.no_grad(): 
                x = super_net(input_var)

            output_var = torch.zeros(x.shape, device = torch.device(my_rank)).detach()

            for i  in range(n_trials): 
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                super_net.reset_binary_gates()  # random sample binary gates
                super_net.unused_modules_off()  # remove unused module for speedup
            
                torch.cuda.synchronize()
                start.record() 

                x = input_var.detach()
                
                # Super Net
                x = super_net(x)
                loss = criterion(x, output_var) 
                super_net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()
                optimizer.step() 
                super_net.unused_modules_back()
                end.record()
                torch.cuda.synchronize()
                if i > n_trials/10: 
                    elapsed_time = start.elapsed_time(end)
                    res.append(elapsed_time) 
                    #for block in super_net.blocks:
                    #    for cand in block:
                    #        print(cand.mobile_inverted_conv.active_index, end = ',')
                    #print(' => %.4f'%(elapsed_time))

            input_var = output_var
            for i in range(int(n_trials/10)):
                res.remove(max(res))
                res.remove(min(res))
        except RuntimeError as e:
            print(e)
            super_net = None 
            torch.cuda.empty_cache()
            input_var = output_var
            res = [87654321]

        s_profs.append(sum(res)/len(res))
        st_profs = []
        for t, s in zip(t_profs, s_profs):
            st_profs.append(t + s)
    
    t_profs =  torch.cuda.FloatTensor(t_profs)
    s_profs =  torch.cuda.FloatTensor(s_profs)
    st_profs =  torch.cuda.FloatTensor(st_profs)
    
    batch_to_t_profs = [torch.zeros_like(t_profs) for _ in range(world_size)] 
    batch_to_s_profs = [torch.zeros_like(s_profs) for _ in range(world_size)] 
    batch_to_st_profs = [torch.zeros_like(st_profs) for _ in range(world_size)] 
    
    print('[',my_rank,'t_profs=', ["%.4f"%(i) for i in t_profs])  
    #print('[',my_rank,'s_profs=', ["%.4f"%(i) for i in s_profs])  
    print('[',my_rank,'st_profs=', ["%.4f"%(i) for i in st_profs])  

    torch.distributed.all_gather(batch_to_t_profs, t_profs)
    torch.distributed.all_gather(batch_to_s_profs, s_profs)
    torch.distributed.all_gather(batch_to_st_profs, st_profs)

    batch_to_t_profs = [x.tolist() for x in batch_to_t_profs]
    batch_to_s_profs = [x.tolist() for x in batch_to_s_profs]
    batch_to_st_profs = [x.tolist() for x in batch_to_st_profs]


    return batch_to_t_profs, batch_to_s_profs, batch_to_st_profs

def measure_comm(args):

    my_rank = int(torch.distributed.get_rank())
    world_size  = int(torch.distributed.get_world_size())
    batch_size = args.train_batch_size
    n_trials = 10

    print("measure_comm start at rank", my_rank)

    # P2P comm
    p2p_profs = []
    if args.dataset == 'imagenet':
        feature_cand = [(batch_size, 24, 56, 56), (batch_size, 32, 28, 28), (batch_size, 64, 14, 14), (batch_size, 96, 14, 14), (batch_size, 160, 7, 7)]
    elif args.dataset == 'cifar10':
        feature_cand = [(batch_size, 24, 32, 32), (batch_size, 32, 16, 16), (batch_size, 64, 8, 8), (batch_size, 96, 8, 8), (batch_size, 160, 4, 4)]
    
    for cand in feature_cand:
        torch.distributed.barrier()
        if world_size > 1 and (my_rank in range(2)):
            res = [] 
            tensor1 = torch.zeros(cand, device= torch.device(my_rank))
            #print(tensor1.element_size() * tensor1.nelement())     

            for i in range(n_trials):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()

                if my_rank == 0:
                    torch.distributed.send(tensor1, dst= 1)
                else:
                    torch.distributed.recv(tensor1, src= 0)

                end.record()
                torch.cuda.synchronize()
                if i>= n_trials/ 10:
                    res.append(start.elapsed_time(end))
            p2p_profs.append(sum(res)/len(res))
        else:
            p2p_profs.append(0)
    
    # DP comm.
    n_blocks = 6
    if args.dataset == 'cifar10':
        n_classes = 10
    elif args.dataset == 'imagenet':
        n_classes = 1000


    dp_profs = []
    if my_rank == 0:
        # No DP comm
        dp_profs = [0 for _ in range(n_blocks)]

    for num_gpus in range(2, world_size+1):
        if num_gpus == 1:
            cur_pgroup = None
        else:
            cur_pgroup = torch.distributed.new_group(list(range(0, num_gpus)), backend ='nccl')
        for block_idx in range(n_blocks):
            torch.distributed.barrier()
            if my_rank in range(0, num_gpus):
                res = []
                super_net = SuperProxylessNASNets(
                        width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
                        conv_candidates=args.conv_candidates, n_classes=n_classes, width_mult=args.width_mult,
                        bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
                    )
                super_net.unused_stages_off(block_idx, block_idx)
                torch.cuda.empty_cache()
                super_net.to(torch.device(my_rank)) 
                MixedEdge.MODE = None
                super_net.init_arch_params(
                    args.arch_init_type, args.arch_init_ratio,
                )
                super_net.unused_modules_back()

                for i  in range(n_trials): 
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start.record() 
                        
                    grad = list() 
                    for param in super_net.weight_parameters():
                        if param is not None and (len(param.size()) > 0):
                            grad.append(param.data) 
                    for param in super_net.binary_gates():
                        if (param is not None) and (len(param.size()) > 0):
                            grad.append(param.data)
                    for param in super_net.architecture_parameters():
                        if (param is not None) and (len(param.size()) > 0):
                            grad.append(param.data)

                    buf =TensorBuffer(grad)
                    #print(buf.buffer.element_size() * buf.buffer.nelement())     
                    torch.distributed.all_reduce(buf.buffer, group = cur_pgroup)
                    
                    end.record()
                    torch.cuda.synchronize()

                    if i > n_trials/10: 
                        elapsed_time = start.elapsed_time(end)
                        res.append(elapsed_time) 
                for i in range(int(n_trials/10)):
                    res.remove(max(res))
                    res.remove(min(res))
                if my_rank == num_gpus - 1:
                    dp_profs.append(sum(res)/len(res))
    
    p2p_profs =  torch.cuda.FloatTensor(p2p_profs)
    dp_profs =  torch.cuda.FloatTensor(dp_profs)
    group_size_to_dp_profs = [torch.zeros_like(dp_profs) for _ in range(world_size)] 
    
    torch.distributed.all_reduce(p2p_profs, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_gather(group_size_to_dp_profs, dp_profs)

    p2p_profs /= 2 # one-to-one comm.
    if my_rank == 0:
        print('[p2p communication profile', ["%.4f"%(i) for i in p2p_profs])
        for group_size, dp_profs in enumerate(group_size_to_dp_profs):
            print('[dp group_size=',group_size+1, 'communication profile', ["%.4f"%(i) for i in dp_profs])  

    return p2p_profs, group_size_to_dp_profs

def schedule(batch_to_st_profs, c_profs, group_size_to_dp_profs, args):
    my_rank = int(torch.distributed.get_rank())
    world_size = int(torch.distributed.get_world_size())
    num_block = 6
    if world_size == 1:
        chosen_dp_mp =  [ 1 ]
        chosen_split_info =  [ 0, num_block]
    elif world_size == 2:
        # 1 + 1 + 1 + 1
        raise NotImplementedError

    elif world_size == 3:
        raise NotImplementedError

    elif world_size == 4:
        min_val = 987654321 
        # 1 + 1 + 1 + 1
        #[(0), (1), (2), (3,4,5)] -> 1, 2, 3
        #[(0), (1), (2,3), (4,5)] -> 1, 2, 4
        #[(0), (1), (2,3,4), (5)] -> 1, 2, 5
        #[(0), (1,2), (3), (4,5)] -> 1, 3, 4
        #[(0), (1,2,),(3,4), (5)] -> 1, 3, 5
        #[(0), (1,2,3), (4), (5)] -> 1, 4, 5
        #[(0,1), (2), (3), (4,5)] -> 2, 3, 4
        #[(0,1), (2), (3,4), (5)] -> 2, 3, 5
        #[(0,1), (2,3), (4), (5)] -> 2, 4, 5
        #[(0,1,2), (3), (4), (5)] -> 3, 4, 5
        for left_split_idx in range(1, num_block-2):
            for mid_split_idx in range(left_split_idx + 1, num_block - 1):
                for right_split_idx in range( mid_split_idx + 1, num_block):
                    left_val = 0 
                    for left in range(0, left_split_idx):
                        left_val += batch_to_st_profs[0][left]

                    mid_val_1 = 0
                    for mid_1 in range(left_split_idx, mid_split_idx):
                        mid_val_1 += batch_to_st_profs[0][mid_1]
                    # p2p comm.
                    mid_val_1 += c_profs[left_split_idx-1]
                    
                    mid_val_2 = 0
                    for mid_2 in range(mid_split_idx, right_split_idx):
                        mid_val_2 += batch_to_st_profs[0][mid_2]
                    mid_val_2 += c_profs[mid_split_idx-1]

                    right_val = 0 
                    for right in range(right_split_idx, num_block):
                        right_val += batch_to_st_profs[0][right]
                    right_val += c_profs[right_split_idx-1] 

                    max_val = mid_val_1  if mid_val_1 > left_val else left_val
                    max_val = max_val  if max_val > mid_val_2 else mid_val_2
                    max_val = max_val  if max_val > right_val else right_val

                    if max_val < min_val:
                        min_val = max_val
                        chosen_split_info = [0, left_split_idx, mid_split_idx, right_split_idx, num_block]
                        chosen_dp_mp = [1, 1, 1, 1]

                        if my_rank == 0:
                            print(chosen_split_info, chosen_dp_mp, "%.4f"%min_val)

        if args.scheme == 'ahd' or args.scheme == 'ckpt':
            # 4
            max_val = 0
            for i in range(num_block):
                max_val += batch_to_st_profs[3][i]
                max_val += group_size_to_dp_profs[3][i]
            if max_val < min_val:
                min_val = max_val
                chosen_split_info = [0, num_block]
                chosen_dp_mp = [4]

                if my_rank == 0:
                    print(chosen_split_info, chosen_dp_mp, "%.4f"%min_val)
            if args.scheme == 'ahd':
                # 3+1 or 2 + 2 or 1 + 3 -> 3이면 batch /3
                #[(0),(1, 2, 3,4,5)] -> split_idx = 0, 1
                #[(0, 1), (2, 3,4,5)] -> split_idx = 2
                #[(0, 1, 2), (3,4,5)] -> split_idx = 3
                #[(0, 1, 2, 3), (4,5)] -> split_idx = 4
                #[(0, 1, 2, 3, 4),(5)] -> split_idx = 5
                #for a, b in [(2,2)]:
                for a, b in [(3,1),(2,2),(1,3)]:
                    for split_idx in range(1, num_block):
                        left_val = 0 
                        for left in range(0, split_idx):
                            left_val += batch_to_st_profs[a-1][left]
                            #DP comm
                            left_val += group_size_to_dp_profs[a-1][left]

                        right_val = 0 
                        for right in range(split_idx, num_block):
                            right_val += batch_to_st_profs[b-1][right]
                            #DP comm
                            right_val += group_size_to_dp_profs[b-1][right]
                        
                        right_val += c_profs[split_idx-1]

                        max_val = right_val  if right_val > left_val else left_val

                        if max_val < min_val:
                            min_val = max_val
                            chosen_split_info = [0, split_idx, num_block]
                            chosen_dp_mp = [a, b]

                            if my_rank == 0:
                                print(chosen_split_info, chosen_dp_mp, "%.4f"%min_val)
                # 2 + 1 + 1 or 1 + 2 + 1 or 1 + 1 + 2
                #[(0),(1),(2, 3,4,5)] -> 1, 2
                #[(0),(1,2), (3,4,5)] -> 1, 3
                #[(0), (1,2,3), (4,5)] -> 1,4
                #[(0), (1,2,3,4), (5)] -> 1,5
                #[(0,1), (2), (3,4,5)] -> 2,3
                #[(0,1), (2,3), (4,5)] -> 2,4
                #[(0,1), (2,3,4), (5)] -> 2,5
                #[(0,1,2), (3), (4,5)] -> 3,4
                #[(0,1,2,), (3,4), (5)] ->3,5
                #[(0,1,2,3), (4), (5)]  -> 4,5
                for a, b, c in [(2,1,1), (1,2,1), (1,1,2)]:
                    for left_split_idx in range(1, num_block-1):
                        for right_split_idx in range( left_split_idx + 1, num_block):
                            left_val = 0 
                            for left in range(0, left_split_idx):
                                left_val += batch_to_st_profs[a-1][left]
                                left_val += group_size_to_dp_profs[a-1][left]


                            mid_val = 0
                            for mid in range(left_split_idx, right_split_idx):
                                mid_val += batch_to_st_profs[b-1][mid]
                                mid_val += group_size_to_dp_profs[b-1][mid]
                            mid_val += c_profs[left_split_idx-1] 

                            right_val = 0 
                            for right in range(right_split_idx,num_block):
                                right_val += batch_to_st_profs[c-1][right]
                                right_val += group_size_to_dp_profs[c-1][right]
                            right_val += c_profs[right_split_idx-1]

                            max_val = mid_val  if mid_val > left_val else left_val
                            max_val = max_val  if max_val > right_val else right_val

                            if max_val < min_val:
                                min_val = max_val
                                chosen_split_info = [0, left_split_idx, right_split_idx, num_block]
                                chosen_dp_mp = [a, b, c]

                                if my_rank == 0:
                                    print(chosen_split_info, chosen_dp_mp, "%.4f"%min_val)
        # chosen_dp_mp
        # [4 ]
        # [(3,1), (2,2), (1,3)]
        # [(2,1,1), (1,2,1), (1,1,2)]
        # [1,1,1,1,]
        
    stage_info = []
    for idx, num_gpu in enumerate(chosen_dp_mp):
        for _ in range(num_gpu):
            stage_info.append((chosen_split_info[idx], chosen_split_info[idx+1] - 1))
    
    if 3 in chosen_dp_mp:
        args.train_batch_size = int(args.train_batch_size / 3)  * 3

    rank_to_batch = []
    for idx, num_gpu in enumerate(chosen_dp_mp):
        for _ in range(num_gpu):
            rank_to_batch.append(int(args.train_batch_size / num_gpu))

    cur_rank = 0
    rank_to_pgroup = []
    debug_pgroup_info = []
    for idx, num_gpu in enumerate(chosen_dp_mp):
        pgroup = torch.distributed.new_group(list(range(cur_rank, cur_rank+num_gpu)),timeout=datetime.timedelta(seconds=120), backend ='nccl')
        for _ in range(num_gpu):
            rank_to_pgroup.append(pgroup)
            debug_pgroup_info.append(list(range(cur_rank, cur_rank+num_gpu)))
        cur_rank += num_gpu
    
    print('pgroup_info:', debug_pgroup_info)
    return stage_info, chosen_dp_mp, rank_to_batch, rank_to_pgroup
         

def check_teacher(args, run_config):
    if args.dataset == 'cifar10': 
        width_stages = '24,32,64,96,160,320'
        n_cell_stages= '2,3,4,3,3,1'
        stride_stages= '1,2,2,1,2,1'

        width_stages =  [int(val) for val in width_stages.split(',')]
        n_cell_stages = [int(val) for val in n_cell_stages.split(',')]
        stride_stages = [int(val) for val in stride_stages.split(',')]

        teacher_net = MobileProxylessNASNets(
            width_stages= width_stages, n_cell_stages= n_cell_stages, stride_stages= stride_stages,
            n_classes= 10,
            bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout,
        )
        
        pretrain_path = 'teachers/cifar10'

        args.valid_size = 5000
        teacher_manager = RunManager(pretrain_path, teacher_net, run_config, out_log = True if my_rank == 0 else False )
        teacher_manager.load_model()
        
        # load checkpoints
        print('Test on test set')
        loss, acc1, acc5 = teacher_manager.validate(is_test=False, return_top5=True)
        log = 'valid_loss: %f\t valid_acc1: %f\t valid_acc5: %f' % (loss, acc1, acc5)
        print(log)

        if acc1 < 90.0:
            raise NotImplementedError

    elif args.dataset == 'imagenet': 
        teacher_net =  build_imagenet_teacher(args, run_config)
        teacher_path = os.path.join(args.path, 'teacher')
        os.makedirs(teacher_path, exist_ok=True)
        
        teacher_manager = RunManager(teacher_path, teacher_net, run_config, out_log = True if my_rank == 0 else False )
        
        print('Test on validation set')
        loss, acc1, acc5 = teacher_manager.validate(is_test=False, return_top5=True)
        log = 'valid_loss: %f\t valid_acc1: %f\t valid_acc5: %f' % (loss, acc1, acc5)
        print(log)

        if acc1 < 70.0:
            raise NotImplementedError


def build_cifar10_teacher(args, run_config):

    width_stages = '24,32,64,96,160,320'
    n_cell_stages= '2,3,4,3,3,1'
    stride_stages= '1,2,2,1,2,1'

    width_stages =  [int(val) for val in width_stages.split(',')]
    n_cell_stages = [int(val) for val in n_cell_stages.split(',')]
    stride_stages = [int(val) for val in stride_stages.split(',')]

    teacher_net = MobileProxylessNASNets(
        width_stages= width_stages, n_cell_stages= n_cell_stages, stride_stages= stride_stages,
        n_classes= 10,
        bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout,
    )
    
    pretrain_path = 'teachers/cifar10'
    
    args.valid_size = 5000
    teacher_manager  = RunManager(pretrain_path, teacher_net, run_config, out_log = False )
    teacher_manager.load_model()

    teacher_manager = None 
    teacher_net = teacher_net.cpu()

    return teacher_net



def build_imagenet_teacher(args, run_config):
    teacher_net = create_model(
        'mobilenetv2_100',
        pretrained=True,
    )
    teacher_net.first_block = teacher_net.blocks[0]
    teacher_net.blocks = teacher_net.blocks[1:]
    
    return teacher_net

def get_hybrid_feature_shape(args, teacher_net, run_config, stage_info, rank_to_batch):
    rank_to_feature_shape = []

    data_shape = [1] + list(run_config.data_provider.data_shape)
    x = torch.zeros(data_shape)
    
    y = x.repeat(rank_to_batch[0],1,1,1).shape # shape has to be contain train, valid tensor
    for s, _ in stage_info:
        if s == 0:
            rank_to_feature_shape.append(y)

    net = teacher_net

    if run_config.dataset == 'cifar10' or run_config.dataset == 'cifar100':
        x = net.first_conv(x)
    elif run_config.dataset == 'imagenet':
        x = net.conv_stem(x)
        x = net.first_block(x)
    else:
        raise NotImplementedError

        
    for idx, block in enumerate(net.blocks):
        x= block(x)
        for rank ,(s, _) in enumerate(stage_info):
            if s == (idx + 1):
                y = x.repeat(rank_to_batch[rank],1,1,1).shape # shape has to be contain train, valid tensor
                rank_to_feature_shape.append(y)

    return rank_to_feature_shape


def get_feature_shape(args, teacher_net, run_config, stage_info):
    block_idx_before_stage = [a for a,_ in stage_info]
    rank_to_feature_shape = []

    data_shape = [1] + list(run_config.data_provider.data_shape)
    x = torch.zeros(data_shape)

    y = x.repeat(args.train_batch_size,1,1,1).shape # shape has to be contain train, valid tensor
    rank_to_feature_shape.append(y)

    net = teacher_net

    if run_config.dataset == 'cifar10' or run_config.dataset == 'cifar100':
        x = net.first_conv(x)
    elif run_config.dataset == 'imagenet':
        x = net.conv_stem(x)
        x = net.first_block(x)
    else:
        raise NotImplementedError

        
    for idx, block in enumerate(net.blocks):
        x= block(x)
        if (idx+1) in block_idx_before_stage:
            y = x.repeat(args.train_batch_size,1,1,1).shape # shape has to be contain train, valid tensor
            rank_to_feature_shape.append(y)

    return rank_to_feature_shape

def load_full_student(args, stage_info, arch_search_config, run_config, rank_to_in_feature_shape):
    first_conv = None
    blocks = []

    for stage_idx, (stage_start, stage_end) in enumerate(stage_info):
        if stage_idx > 0 and stage_info[stage_idx - 1][0] == stage_start:
            continue
        stage_path = os.path.join(args.path, str(stage_idx))
        
        # Empty super net
        super_net = SuperProxylessNASNets(
            width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
            conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
            bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
        )
        super_net.unused_stages_off(stage_start, stage_end)
        arch_search_run_manager = ArchSearchRunManager(stage_path, super_net, run_config, arch_search_config, False, stage_idx, rank_to_in_feature_shape, None)
        arch_search_run_manager.load_model()
        
        partial_net = arch_search_run_manager.run_manager.net

        if partial_net.first_conv is not None:
            first_conv = copy.deepcopy(partial_net.first_conv)
        for block in partial_net.blocks:
            blocks.append(copy.deepcopy(block))
        if partial_net.feature_mix_layer is not None:
            feature_mix_layer = copy.deepcopy(partial_net.feature_mix_layer)
            classifier = copy.deepcopy(partial_net.classifier)
        
        super_net = None
        arch_search_run_manager=None
        torch.cuda.empty_cache()

    assert(first_conv is not None)
    assert(len(blocks) == len(args.n_cell_stages)) 
    assert(feature_mix_layer is not None)
    assert(classifier is not None)

    rank_path = os.path.join(args.path, str(torch.distributed.get_rank()))
    super_net = SuperProxylessNASNets(
        width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
        conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
        bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
    )
    super_net.first_conv = first_conv
    super_net.blocks = torch.nn.ModuleList(blocks)
    super_net.feature_mix_layer = feature_mix_layer
    super_net.classifier = classifier
    
    return super_net



if __name__ == '__main__':
    args = parser.parse_args()
    
    #os.environ["PL_GLOBAL_SEED"] = str(args.manual_seed)
    #os.environ["PYTHONHASHSEED"] = str(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)
    #torch.backends.cudnn.deterministic =True
    #torch.backends.cudnn.benchmark = False
    
    os.makedirs(args.path, exist_ok=True)
    
    width_stages_str = '-'.join(args.width_stages.split(','))
    # build net from args
    args.width_stages = [int(val) for val in args.width_stages.split(',')]
    args.n_cell_stages = [int(val) for val in args.n_cell_stages.split(',')]
    args.stride_stages = [int(val) for val in args.stride_stages.split(',')]
    args.conv_candidates = [
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
    ]

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #os.environ['NCCL_BLOCKING_WAIT'] = 1 
    process_group = torch.distributed.init_process_group(
        backend = "nccl",
        timeout= datetime.timedelta(seconds=987654),
        world_size = int(os.environ["WORLD_SIZE"]),
        rank = int(os.environ["LOCAL_RANK"]),
    )
    my_rank = torch.distributed.get_rank()
    world_size = int(os.environ["WORLD_SIZE"]) 
    args.num_replicas =  int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(my_rank)
    if args.dataset == 'cifar10':
        args.valid_size = 5000
        build_teacher = build_cifar10_teacher
        if args.target_hardware is None:
            args.ref_value = None
        elif args.target_hardware == 'flops':
            args.ref_value = 88 * 1e6 # ProxylessNAS 
        else:
            raise NotImplementedError
    elif args.dataset == 'imagenet':
        args.valid_size = 50000
        build_teacher = build_imagenet_teacher
        if args.target_hardware is None:
            args.ref_value = None
        else:
            args.ref_value = ref_values[args.target_hardware]['%.2f' % args.width_mult]
    else:
        raise NotImplementedError

    if args.train_method == 'ts': 
        if args.dataset == 'cifar10':
            run_config = Cifar10RunConfig(
                **args.__dict__
            )
        elif args.dataset == 'imagenet':
            run_config = ImagenetRunConfig(
                **args.__dict__
            )

        #batch_to_t_profs, batch_to_s_profs, batch_to_st_profs = profile(args, run_config= run_config)
        torch.distributed.barrier()

    elif args.train_method == 'mp':
        if args.dataset == 'cifar10':
            run_config = Cifar10RunConfig(
                **args.__dict__
            )
        elif args.dataset == 'imagenet':
            run_config = ImagenetRunConfig(
                **args.__dict__
            )
        
        batch_to_t_profs, batch_to_s_profs, batch_to_st_profs = profile(args, run_config= run_config)
        
        torch.distributed.barrier() 
        c_profs, dp_profs = measure_comm(args)
        torch.distributed.barrier() 

        stage_info, chosen_dp_mp, rank_to_batch, rank_to_pgroup =  schedule(batch_to_st_profs, c_profs, dp_profs, args)
    
        if my_rank == 0:
            print('stage_info', stage_info)
            print('rank_to_batch', rank_to_batch)
            print('chosen_dp_mp', chosen_dp_mp)

        args.train_batch_size = (int(args.train_batch_size /chosen_dp_mp[0]) * int(chosen_dp_mp[0]))
        if my_rank in range(chosen_dp_mp[0]): 
            args.train_batch_size /=  int(chosen_dp_mp[0])
            args.train_batch_size =  int(args.train_batch_size)
        
        args.num_replicas = chosen_dp_mp[0]
        args.n_worker = int(args.n_worker / chosen_dp_mp[0])
          
        torch.distributed.barrier()
    else:
        args.train_batch_size /=  int(os.environ["WORLD_SIZE"])
        args.train_batch_size =  int(args.train_batch_size)
        args.test_batch_size /=  int(os.environ["WORLD_SIZE"])
        args.test_batch_size =  int(args.test_batch_size)
        args.n_worker = int(args.n_worker / int(os.environ["WORLD_SIZE"]))

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    if args.dataset == 'cifar10':
        args.valid_size = 5000
        run_config = Cifar10RunConfig(
            **args.__dict__
        )
        build_teacher = build_cifar10_teacher

        if args.target_hardware is None:
            args.ref_value = None
        elif args.target_hardware == 'flops':
            args.ref_value = 88 * 1e6 # ProxylessNAS 
        else:
            raise NotImplementedError

    elif args.dataset == 'imagenet':
        args.valid_size = 50000
        run_config = ImagenetRunConfig(
            **args.__dict__
        )
        build_teacher = build_imagenet_teacher
        if args.target_hardware is None:
            args.ref_value = None
        else:
            args.ref_value = ref_values[args.target_hardware]['%.2f' % args.width_mult]
    else:
        raise NotImplementedError

    # build arch search config from args
    if args.arch_opt_type == 'adam':
        args.arch_opt_param = {
            'betas': (args.arch_adam_beta1, args.arch_adam_beta2),
            'eps': args.arch_adam_eps,
        }
    else:
        args.arch_opt_param = None

    
    if args.arch_algo == 'grad':
        from nas_manager import GradientArchSearchConfig
        if args.grad_reg_loss_type == 'add#linear':
            args.grad_reg_loss_params = {'lambda': args.grad_reg_loss_lambda}
        elif args.grad_reg_loss_type == 'mul#log':
            args.grad_reg_loss_params = {
                'alpha': args.grad_reg_loss_alpha,
                'beta': args.grad_reg_loss_beta,
            }
        else:
            args.grad_reg_loss_params = None
        arch_search_config = GradientArchSearchConfig(**args.__dict__)
    elif args.arch_algo == 'rl':
        from nas_manager import RLArchSearchConfig
        arch_search_config = RLArchSearchConfig(**args.__dict__)
    else:
        raise NotImplementedError
    
    
        
    out_log = False
    if my_rank ==0 :
        out_log =True
        print('Run config:')
        for k, v in run_config.config.items():
            print('\t%s: %s' % (k, v))
        print('Architecture Search config:')
        for k, v in arch_search_config.config.items():
            print('\t%s: %s' % (k, v))

    if  args.train_method == 'ts':
        #check_teacher(args, run_config)
        rank_to_stage_info = [[(0,0)], [(1,1)],[(2,2),(3,3)],[(4,4),(5,5)]]
        stage_info = rank_to_stage_info[my_rank] 
        teacher_net = build_teacher(args, run_config)
        rank_to_in_feature_shape = get_feature_shape(args, teacher_net, run_config, stage_info)
        
        st = time.time()
        for stage_start, stage_end in stage_info:
            teacher_net = build_teacher(args, run_config)
            super_net = SuperProxylessNASNets(
                width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
                conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
                bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
            )
            teacher_net.unused_stages_off(0, stage_end)
            super_net.unused_stages_off(stage_start, stage_end)
            torch.cuda.empty_cache()

            stage_path = os.path.join(args.path, str(stage_start))
            os.makedirs(stage_path, exist_ok=True)
          
            arch_search_run_manager = ArchSearchRunManager(stage_path, super_net, run_config, arch_search_config, True, stage_start, rank_to_in_feature_shape, teacher_net= teacher_net)
            torch.cuda.empty_cache()
           
            arch_search_run_manager.train_stage(stage_start, len(stage_info), fix_net_weights=args.debug)

            teacher_net = None
            super_net = None
            arch_search_run_manager = None
            torch.cuda.empty_cache()

        print('[Rank at %d] Total elapsed time: %.4f'%(my_rank, time.time()- st))
        #time.sleep(360) 
        torch.distributed.barrier()

        stage_info = [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5)]
    elif args.train_method == 'dp': 
        super_net = SuperProxylessNASNets(
            width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
            conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
            bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
        )
        # arch search run manager
        arch_search_run_manager = ArchSearchRunManager(args.path, super_net, run_config, arch_search_config, out_log = out_log)

        # resume
        if args.resume:
            try:
                arch_search_run_manager.load_model()
            except Exception:
                from pathlib import Path
                home = str(Path.home())
                warmup_path = os.path.join(
                    home, 'Workspace/Exp/arch_search/%s_ProxylessNAS_%.2f_%s/warmup.pth.tar' %
                          (run_config.dataset, args.width_mult, width_stages_str)
                )
                if os.path.exists(warmup_path):
                    print('load warmup weights')
                    arch_search_run_manager.load_model(model_fname=warmup_path)
                else:
                    print('fail to load models')

        # warmup
        if arch_search_run_manager.warmup:
            arch_search_run_manager.warm_up(warmup_epochs=args.warmup_epochs)

        # joint training
        arch_search_run_manager.train(fix_net_weights=args.debug)
    elif args.train_method == 'dna':
        #check_teacher(args, run_config)
        stage_info = [(0,0), (1,1), (2,2),(3,3),(4,4),(5,5)]
        
        teacher_net = build_teacher(args, run_config)
        rank_to_in_feature_shape = get_feature_shape(args, teacher_net, run_config, stage_info)
        if args.target_hardware == 'flops':
            data_shape = list(rank_to_in_feature_shape[0])
            data_shape[0] = 1 # remove batch size 
            input_var = torch.zeros(data_shape)
            with torch.no_grad():
                if args.dataset == 'imagenet':
                    ref_teacher_net = MobileProxylessNASNets(
                        width_stages=args.width_stages, n_cell_stages=[2,3,4,3,3,1], stride_stages=args.stride_stages,
                        n_classes= 1000,
                    )
                elif args.dataset == 'cifar10':
                    ref_teacher_net = MobileProxylessNASNets(
                        width_stages=args.width_stages, n_cell_stages=[2,3,4,3,3,1], stride_stages=args.stride_stages,
                        n_classes= 10,
                    )
            teacher_flops, _ = ref_teacher_net.get_flops(input_var)
            if my_rank == 0:
                print('teacher flops:%.2fM'%(teacher_flops /1e6))
        
        for stage_idx, (stage_start, stage_end) in enumerate(stage_info):
            teacher_net = build_teacher(args, run_config)
            super_net = SuperProxylessNASNets(
                width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
                conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
                bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
            )
            teacher_net.unused_stages_off(0, stage_end)
            super_net.unused_stages_off(stage_start, stage_end)
            #with torch.no_grad():
            #    if my_rank == 0: 
            #        summary(teacher_net, rank_to_in_feature_shape[0][1:], batch_size = args.train_batch_size, device = torch.device('cpu').type)
            #        summary(super_net, rank_to_in_feature_shape[stage_idx][1:], batch_size = args.train_batch_size, device= torch.device('cpu').type)
            #        print(count_parameters(super_net))

            torch.distributed.barrier()

            stage_path = os.path.join(args.path, str(stage_idx))
            os.makedirs(stage_path, exist_ok=True)
          
            if my_rank == 0:
                print('\n', '-' * 30, 'Train stage: %d' % (stage_idx + 1), '-' * 30, '\n')
            """ flops option """
            if args.target_hardware == 'flops':
                with torch.no_grad():
                    if args.dataset == 'imagenet':
                        ref_teacher_net = MobileProxylessNASNets(
                            width_stages=args.width_stages, n_cell_stages=[2,3,4,3,3,1], stride_stages=args.stride_stages,
                            n_classes= 1000,
                        )
                    elif args.dataset == 'cifar10':
                        ref_teacher_net = MobileProxylessNASNets(
                            width_stages=args.width_stages, n_cell_stages=[2,3,4,3,3,1], stride_stages=args.stride_stages,
                            n_classes= 10,
                        )
                    ref_teacher_net.unused_stages_off(stage_start, stage_end)

                    data_shape = list(rank_to_in_feature_shape[stage_idx])
                    data_shape[0] = 1 # remove batch size 
                    input_var = torch.zeros(data_shape)
                    
                    teacher_flops, _ = ref_teacher_net.get_flops(input_var)

                if my_rank == 0:
                    print('teacher flops:%.2fM'%(teacher_flops /1e6))
            
                arch_search_config.ref_value = int(teacher_flops) # ProxylessNAS -> 0.5 / 0.645
            arch_search_run_manager = ArchSearchRunManager(stage_path, super_net, run_config, arch_search_config, out_log, stage_idx, rank_to_in_feature_shape, teacher_net= teacher_net)
            torch.cuda.empty_cache()
           
            torch.distributed.barrier()
            arch_search_run_manager.train_stage(stage_idx, len(stage_info), fix_net_weights=args.debug)

            teacher_net = None
            super_net = None
            arch_search_run_manager = None
            torch.cuda.empty_cache()
            if my_rank == 0:
                print("allocated after free: %.2fM"%(torch.cuda.memory_allocated()/1e6))
    elif args.train_method == 'mp':
        #check_teacher(args, run_config)
        stage_path = os.path.join(args.path, str(my_rank))
        os.makedirs(stage_path, exist_ok=True)
        
        start, end  = stage_info[my_rank]
        """ arch search run manager """

        teacher_net = build_teacher(args, run_config)
        rank_to_in_feature_shape = get_hybrid_feature_shape(args, teacher_net, run_config, stage_info, rank_to_batch)
        super_net = SuperProxylessNASNets(
            width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
            conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
            bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
        )

        teacher_net.unused_stages_off(start, end )
        super_net.unused_stages_off(start, end )
                       
        """ flops option """
        if args.target_hardware == 'flops':
            arch_search_config.ref_value = list()
            data_shape = list(rank_to_in_feature_shape[my_rank])
            data_shape[0] = 1 # remove batch size 

            for block_idx in range(start, end+1):
                with torch.no_grad():
                    if args.dataset == 'imagenet':
                        ref_teacher_net = MobileProxylessNASNets(
                            width_stages=args.width_stages, n_cell_stages=[2,3,4,3,3,1], stride_stages=args.stride_stages,
                            n_classes= 1000,
                        )
                    elif args.dataset == 'cifar10':
                        ref_teacher_net = MobileProxylessNASNets(
                            width_stages=args.width_stages, n_cell_stages=[2,3,4,3,3,1], stride_stages=args.stride_stages,
                            n_classes= 10,
                        )
                    ref_teacher_net.unused_stages_off(block_idx, block_idx)

                    input_var = torch.zeros(data_shape)
                     
                    teacher_flops, out_feature = ref_teacher_net.get_flops(input_var)
                    data_shape = list(out_feature.shape)

                arch_search_config.ref_value.append(int(teacher_flops)) # ProxylessNAS -> 0.5 / 0.645
        torch.distributed.barrier()

        arch_search_run_manager = ArchSearchRunManager(stage_path, super_net, run_config, arch_search_config, out_log, my_rank, rank_to_in_feature_shape, rank_to_pgroup, teacher_net)

        arch_search_run_manager.train_mp(my_rank, rank_to_in_feature_shape, chosen_dp_mp, stage_info, rank_to_batch, rank_to_pgroup, args.scheme)
            

                            
    else:
        raise NotImplementedError
    
    print('Arch search ends in device %d'%(my_rank))
    torch.cuda.synchronize()
    torch.distributed.barrier()
     
    if args.train_method != 'dp' and my_rank == 0:
        run_config.train_method = None
        first_conv = None
        blocks = []
        for stage_idx, (stage_start, stage_end) in enumerate(stage_info):
            if stage_idx > 0 and stage_info[stage_idx - 1][0] == stage_start:
                continue
            stage_path = os.path.join(args.path, str(stage_idx))
            
            # Empty super net
            super_net = SuperProxylessNASNets(
                width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
                conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
                bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
            )
            print('try to load model to device %d'%(my_rank))

            super_net.unused_stages_off(stage_start, stage_end)

            print('here?')
            arch_search_run_manager = ArchSearchRunManager(stage_path, super_net, run_config, arch_search_config, False, stage_idx, rank_to_in_feature_shape, None)
            print('here?')
            arch_search_run_manager.load_model()
            
            print('sucess in load model to device %d'%(my_rank))
            partial_net = arch_search_run_manager.net

            if partial_net.first_conv is not None:
                first_conv = copy.deepcopy(partial_net.first_conv)
            for block in partial_net.blocks:
                blocks.append(copy.deepcopy(block))
            if partial_net.feature_mix_layer is not None:
                feature_mix_layer = copy.deepcopy(partial_net.feature_mix_layer)
                classifier = copy.deepcopy(partial_net.classifier)
            
            print("allocated for stage", stage_idx, ": %.2fM"% (torch.cuda.memory_allocated()/1e6))
            super_net = None
            arch_search_run_manager=None
            torch.cuda.empty_cache()
            print("allocated after free: %.2fM"%( torch.cuda.memory_allocated()/1e6))

        assert(first_conv is not None)
        assert(len(blocks) == len(args.n_cell_stages)) 
        assert(feature_mix_layer is not None)
        assert(classifier is not None)

        super_net = SuperProxylessNASNets(
            width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
            conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
            bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
        )
        arch_search_run_manager = ArchSearchRunManager(os.path.join(args.path, 'learned_net'), super_net, run_config, arch_search_config, True)
        arch_search_run_manager.run_manager.net.first_conv = first_conv
        arch_search_run_manager.run_manager.net.blocks = torch.nn.ModuleList(blocks)
        arch_search_run_manager.run_manager.net.feature_mix_layer = feature_mix_layer
        arch_search_run_manager.run_manager.net.classifier = classifier

        for stage in arch_search_run_manager.run_manager.net.blocks:
            for idx, block in enumerate(stage):
                arch_search_run_manager.write_log('%d. %s' % (idx, block.module_str), prefix='arch')

        normal_net = super_net.convert_to_normal_net()
        os.makedirs(os.path.join(args.path, 'learned_net'), exist_ok=True)
        
        run_manager = RunManager(os.path.join(args.path, 'learned_net'), normal_net, run_config, out_log= True)
        json.dump(normal_net.config , open(os.path.join(args.path, 'learned_net/net.config'), 'w'), indent=4)
        json.dump(
            run_config.config,
            open(os.path.join(args.path, 'learned_net/run.config'), 'w'), indent=4,
        )
        torch.save(
            {'state_dict': normal_net.state_dict(), 'dataset': run_config.dataset},
            os.path.join(args.path, 'learned_net/init')
        )
        print('Save searched model success!')
    torch.distributed.barrier()

