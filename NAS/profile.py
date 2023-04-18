
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


