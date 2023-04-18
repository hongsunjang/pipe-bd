# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import argparse
import datetime

from models import Cifar10RunConfig
from nas_manager import *
from models.super_nets.super_proxyless import SuperProxylessNASNets

import matplotlib.pyplot as plt

# ref values
ref_values = {
    'flops': {
        '0.35': 59 * 1e6,
        '0.50': 97 * 1e6,
        '0.75': 209 * 1e6,
#        '1.00': 300 * 1e6,
        '1.00': 80 * 1e6,
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
parser.add_argument('--manual_seed', default=0, type=int)

""" run config """
parser.add_argument('--n_epochs', type=int, default=120)
parser.add_argument('--init_lr', type=float, default=0.025)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
# lr_schedule_param

parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=512)
parser.add_argument('--valid_size', type=int, default=5000)

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str, default=None, choices=[None, 'bn', 'bn#bias'])

parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=1)

parser.add_argument('--n_worker', type=int, default=32)
parser.add_argument('--resize_scale', type=float, default=0.08)
parser.add_argument('--distort_color', type=str, default='normal', choices=['normal', 'strong', None])
parser.add_argument('--cutout', type=int, default=None)

""" net config """
#parser.add_argument('--n_cell_stages', type=str, default='4,4,4,4,4,1')

parser.add_argument('--width_stages', type=str, default='24,32,64,96,160,320')
parser.add_argument('--teacher_n_cell_stages', type=str, default='2,3,4,3,3,1')
parser.add_argument('--n_cell_stages', type=str, default='4,4,4,4,4,1')
parser.add_argument('--stride_stages', type=str, default='1,2,2,1,2,1')
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
parser.add_argument('--arch_lr', type=float, default=1e-3)
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
parser.add_argument('--grad_reg_loss_alpha', type=float, default=0.2)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_beta', type=float, default=0.3)  # grad_reg_loss_params
""" RL hyper-parameters """
parser.add_argument('--rl_batch_size', type=int, default=10)
parser.add_argument('--rl_update_per_epoch', action='store_true')
parser.add_argument('--rl_update_steps_per_epoch', type=int, default=300)
parser.add_argument('--rl_baseline_decay_weight', type=float, default=0.99)
parser.add_argument('--rl_tradeoff_ratio', type=float, default=0.1)

parser.add_argument('--train_method', type =str, required= True, choices=['dp', 'dna', 'mp'])
parser.add_argument('--start_stage', type = int, default=0)

def build_teacher(args):
    from build_mobilenet import MobileProxylessNASNets
    teacher_net = MobileProxylessNASNets(
        width_stages=args.width_stages, n_cell_stages=args.teacher_n_cell_stages, stride_stages=args.stride_stages,
        n_classes= 10,
        bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout,
    )
    pretrain_path = 'teachers/5e4'
    model_fname = os.path.join(pretrain_path, 'checkpoint/model_best.pth.tar')
    print('try to load checkpoint from %s ...' % model_fname)
    try:
        checkpoint = torch.load(model_fname, map_location='cpu')
        teacher_net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(model_fname))
    except Exception:
        print('fail to load checkpoint from %s' % pretrain_path)
        raise NotImplementedError

    return teacher_net

def get_feature_shape(args, teacher_net, run_config, stage_info):
    block_idx_before_stage = [a for a,_ in stage_info]
    rank_to_feature_shape = []
    data_shape = [1] + list(run_config.data_provider.data_shape)
    x = torch.zeros(data_shape)
    net = teacher_net
    x = net.first_conv(x)
    #x = net.conv_stem(x)
    for idx, block in enumerate(net.blocks):
        print(x.shape)
        x= block(x)
        if (idx+1) in block_idx_before_stage:
            y = x.repeat(args.train_batch_size,1,1,1).shape # shape has to be contain train, valid tensor
            rank_to_feature_shape.append(y)

    return rank_to_feature_shape


def profile(my_rank, args, run_config):
    b_choices = [args.train_batch_size, int(args.train_batch_size/2),int(args.train_batch_size/3), int(args.train_batch_size /4)]
    
    batch_size  = b_choices[my_rank]
    data_shape = [batch_size] + list(run_config.data_provider.data_shape)
    input_var = torch.zeros(data_shape, device = torch.device(my_rank))
    
    n_blocks = 6
    n_trials = 200

    t_profs = []
    for block_idx in range(n_blocks):
        res = []
        teacher_net = build_teacher(args)
        teacher_net.unused_stages_off(block_idx, block_idx)
        teacher_net.to(torch.device(my_rank)) 
        x = teacher_net(input_var)
        output_var = torch.zeros(x.shape, device = torch.device(my_rank)).detach()

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
        input_var = output_var
        t_profs.append(sum(res)/len(res))
    print('[teacher net profile when batch size =', b_choices[my_rank],']', ["%.4f"%(i) for i in t_profs])
    
    s_profs = []
    input_var = torch.zeros(data_shape, device = torch.device(my_rank))
    for block_idx in range(n_blocks):
        res = []
        super_net = SuperProxylessNASNets(
                width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
                conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
                bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
            )
        super_net.unused_stages_off(block_idx, block_idx)
        super_net.to(torch.device(my_rank)) 
        MixedEdge.MODE = None
        super_net.init_arch_params(
            args.arch_init_type, args.arch_init_ratio,
        )
        super_net.unused_modules_back()
        criterion = nn.MSELoss().to(torch.device(my_rank))
    
        optimizer = torch.optim.SGD(super_net.weight_parameters(), 0.001, momentum=0.9,weight_decay = 0.001, nesterov=True)
        
        x = super_net(input_var)
        output_var = torch.zeros(x.shape, device = torch.device(my_rank)).detach()
        for i in range(n_trials): 
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record() 

            super_net.reset_binary_gates()  # random sample binary gates
            super_net.unused_modules_off()  # remove unused module for speedup

            x = input_var.detach()
            
            # Super Net
            x = super_net(x)
            loss = criterion(x, output_var) 
            super_net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
            loss.backward()
            optimizer.step() 
            torch.cuda.synchronize()
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
        for i in range(n_trials/10):
            res.remove(max(res))
            res.remove(min(res))
        s_profs.append(sum(res)/len(res))
        
        plt.subplot(1,2,1)
        plt.scatter(range(len(res)),res, marker='.') 
        plt.subplot(1,2,2)
        plt.hist(res, bins=50)
        plt.show()
        plt.savefig(os.path.join('images','['+str(b_choices[my_rank])+']'+str(block_idx)+'.png'))
    print('[super net profile at rank ', b_choices[my_rank],']', ["%.4f"%(i) for i in s_profs])

    return t_profs, s_profs


def get_net_flops( stage, arch_search_run_manager, rank_to_feature_shape):
    batch_size = 128
    if stage == 0:
        data_shape = [batch_size] + list(arch_search_run_manager.run_manager.run_config.data_provider.data_shape)
    else:
        data_shape = list(rank_to_feature_shape[stage - 1])
        data_shape[0] = batch_size # remove batch size 
    input_var = torch.zeros(data_shape, device = arch_search_run_manager.run_manager.device)

    # TeacherNet
    with torch.no_grad():
        ref_flops, _ = arch_search_run_manager.teacher_net.get_flops(input_var)

    # SuperNet
    expected_flops = 0
    super_net = arch_search_run_manager.net

    x = input_var
    # first conv
    if super_net.first_conv is not None:
        flop, x = super_net.first_conv.get_flops(x)
        expected_flops += flop

    # blocks
    for stage in super_net.blocks:
        for block in stage:
            mb_conv = block.mobile_inverted_conv
            if not isinstance(mb_conv, MixedEdge):
                delta_flop, x = block.get_flops(x)
                expected_flops = expected_flops + delta_flop
                continue

            if block.shortcut is None:
                shortcut_flop = 0
            else:
                shortcut_flop, _ = block.shortcut.get_flops(x)
            expected_flops = expected_flops + shortcut_flop

            total_op_flops = 0 
            for i, op in enumerate(mb_conv.candidate_ops):
                if op is None or op.is_zero_layer():
                    continue
                op_flops, _ = op.get_flops(x)
                total_op_flops += op_flops 

            expected_flops += (total_op_flops / len(mb_conv.candidate_ops))

            x = block(x)


    # feature mix layer
    if super_net.feature_mix_layer is not None:
        delta_flop, x = super_net.feature_mix_layer.get_flops(x)
        expected_flops += delta_flop
        # classifier
        x = super_net.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        delta_flop, x = super_net.classifier.get_flops(x)
        expected_flops = expected_flops + delta_flop

    return expected_flops, ref_flops


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.makedirs(args.path, exist_ok=True)
   
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    process_group = torch.distributed.init_process_group(
        backend = "nccl",
        timeout= datetime.timedelta(seconds=60),
        world_size = int(os.environ["WORLD_SIZE"]),
        rank = int(os.environ["LOCAL_RANK"]),
    )
    my_rank = torch.distributed.get_rank()
    torch.cuda.set_device(my_rank)

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    run_config = Cifar10RunConfig(
        **args.__dict__
    )

    # debug, adjust run_config
    if args.debug:
        run_config.train_batch_size = 256
        run_config.test_batch_size = 256
        run_config.valid_size = 256
        run_config.n_worker = 0

    width_stages_str = '-'.join(args.width_stages.split(','))
    # build net from args
    args.width_stages = [int(val) for val in args.width_stages.split(',')]
    args.teacher_n_cell_stages=[int(val) for val in args.teacher_n_cell_stages.split(',')]
    args.n_cell_stages = [int(val) for val in args.n_cell_stages.split(',')]
    args.stride_stages = [int(val) for val in args.stride_stages.split(',')]
    args.conv_candidates = [
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
    ]

    # build arch search config from args
    if args.arch_opt_type == 'adam':
        args.arch_opt_param = {
            'betas': (args.arch_adam_beta1, args.arch_adam_beta2),
            'eps': args.arch_adam_eps,
        }
    else:
        args.arch_opt_param = None
    if args.target_hardware is None:
        args.ref_value = None
    else:
        args.ref_value = ref_values[args.target_hardware]['%.2f' % args.width_mult]
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
    if my_rank == 0:
        out_log = True
        print('Run config:')
        for k, v in run_config.config.items():
            print('\t%s: %s' % (k, v))
        json.dump(
            run_config.config,
            open(os.path.join(args.path, 'run.config'), 'w'), indent=4,
        )
        print('Architecture Search config:')
        for k, v in arch_search_config.config.items():
            print('\t%s: %s' % (k, v))
    
    t_profs, s_profs = profile(my_rank, args, run_config)
    torch.cuda.empty_cache()
        
    # Scheduling algorithm here #TODO
    
    
    conf = [4]
    stage_info = [(0,6), (0,6), (0,6), (0,6)]

    if args.train_method == 'dp':
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
        teacher_net = build_teacher(args)
        rank_to_feature_shape = get_feature_shape(args, teacher_net, run_config, stage_info)
        for stage_idx, (stage_start, stage_end) in enumerate(stage_info):
            if stage_idx < args.start_stage:
                continue
            if stage_start == 0 and stage_end == 0:
                continue
            teacher_net = build_teacher(args, n_classes=run_config.data_provider.n_classes)
            super_net = SuperProxylessNASNets(
                width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
                conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
                bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
            )
            #teacher_net.unused_stages_off(0, stage_end,  stage_start)
            #super_net.unused_stages_off(stage_start, stage_end)
            #ref_flops = get_net_flops(teacher_net, stage_idx, run_config, rank_to_feature_shape)
            torch.cuda.empty_cache()

            """ arch search run manager """
            stage_path = os.path.join(args.path, str(stage_idx))
            os.makedirs(stage_path, exist_ok=True)
            arch_search_run_manager = ArchSearchRunManager(stage_path, super_net, run_config, arch_search_config, out_log, stage_idx, rank_to_feature_shape, teacher_net)
            ref_flops_value = [1703936,13221888,24797184,45741202]
            ref_flops = ref_flops_value[stage_idx]
            print('[rank %d]teacher flops: %.2f'%(my_rank, ref_flops/1e6))
            #print('[rank %d]student expected flops: %.2f'%(my_rank, expected_flops))
            arch_search_run_manager.arch_search_config.ref_value = ref_flops
            assert(args.target_hardware == 'flops')

            arch_search_run_manager.train_stage(stage_idx, fix_net_weights=args.debug)
            teacher_net = None
            super_net = None
            arch_search_run_manager = None
            torch.cuda.empty_cache()

    elif args.train_method == 'mp':
        assert torch.distributed.is_available()
        start, end  = stage_info[my_rank]
        """ arch search run manager """
        stage_path = os.path.join(args.path, str(my_rank))
        os.makedirs(stage_path, exist_ok=True)
        
        teacher_net = build_teacher(args, n_classes=run_config.data_provider.n_classes)
        #TODO: rank to feature shape should be change
        rank_to_feature_shape = get_feature_shape(args, teacher_net, run_config, stage_info)
        super_net = SuperProxylessNASNets(
            width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
            conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
            bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
        )
        
        teacher_net.unused_stages_off(start, end)
        super_net.unused_stages_off(start, end)
        

        args.train_batch_size = args.train_batch_size/conf[0]
        run_config = Cifar10RunConfig(
            **args.__dict__
        )

        arch_search_run_manager = ArchSearchRunManager(stage_path, super_net, run_config, arch_search_config, out_log, my_rank, rank_to_feature_shape, teacher_net)
        
        #expected_flops, ref_flops = get_net_flops(my_rank, arch_search_run_manager, rank_to_feature_shape)
        torch.cuda.empty_cache()
        print('[rank %d]teacher flops: %.2f'%(my_rank, ref_flops))
        print('[rank %d]student expected flops: %.2f'%(my_rank, expected_flops))
        arch_search_run_manager.arch_search_config.ref_value = ref_flops
        assert(args.target_hardware == 'flops')

        arch_search_run_manager.train_mp(my_rank, rank_to_feature_shape, fix_net_weights=args.debug)

        teacher_net = None
        super_net= None
        arch_search_run_manager = None
    else:
        raise NotImplementedError

    torch.cuda.empty_cache()
    assert(torch.cuda.memory_allocated(0) == 0)
    assert(torch.cuda.memory_allocated(1) == 0)
    assert(torch.cuda.memory_allocated(2) == 0)
    assert(torch.cuda.memory_allocated(3) == 0)

    if args.train_method != 'dp' and my_rank == 0:
        run_config.train_method = None
        tmp = args.n_cell_stages.copy()
        tmp[0] += 1
        encodings = []
        for i in range(6):
            for j in range(tmp[i]):
                encodings.append(i)

        first_conv = None
        blocks = []
        for stage_idx, (stage_start, stage_end) in enumerate(stage_info):
            stage_path = os.path.join(args.path, str(stage_idx))
            
            # Empty super net
            super_net = SuperProxylessNASNets(
                width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
                conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
                bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
            )
            super_net.unused_stages_off(stage_start, stage_end)
            torch.cuda.empty_cache()
            arch_search_run_manager = ArchSearchRunManager(stage_path, super_net, run_config, arch_search_config, False, stage_idx, rank_to_feature_shape, teacher_net = None)
            if not (stage_start == 0 and stage_end == 0):
                arch_search_run_manager.load_model()
            
            super_net = arch_search_run_manager.net

            if super_net.first_conv is not None:
                first_conv = copy.deepcopy(super_net.first_conv)
            for block in super_net.blocks:
                blocks.append(copy.deepcopy(block))
            if super_net.feature_mix_layer is not None:
                feature_mix_layer = copy.deepcopy(super_net.feature_mix_layer)
                classifier = copy.deepcopy(super_net.classifier)
            
            super_net = None
            arch_search_run_manager = None
            torch.cuda.empty_cache()
            print("allocated after free: %.2fM"%( torch.cuda.memory_allocated()/1e6))


        assert(first_conv is not None)
        assert(len(blocks) == len(args.n_cell_stages)+1) 
        assert(feature_mix_layer is not None)
        assert(classifier is not None)

        super_net = SuperProxylessNASNets(
            width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
            conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
            bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
        )
        arch_search_run_manager = ArchSearchRunManager(os.path.join(args.path, 'learned_net'), super_net, run_config, arch_search_config, out_log = True)
        arch_search_run_manager.net.first_conv = first_conv
        arch_search_run_manager.net.blocks = torch.nn.ModuleList(blocks)
        arch_search_run_manager.net.feature_mix_layer = feature_mix_layer
        arch_search_run_manager.net.classifier = classifier

        for stage in arch_search_run_manager.net.blocks:
            for idx, block in enumerate(stage):
                arch_search_run_manager.write_log('%d. %s' % (idx, block.module_str), prefix='arch')

        normal_net = arch_search_run_manager.net.convert_to_normal_net()
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

