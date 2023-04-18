import torch
from models.super_nets.super_proxyless_cifar10  import *
from data_providers.cifar10 import *

def return_flops(net, x, n_cell_stages):
    flops = torch.zeros(6)
    delta_flop, x = net.first_conv.get_flops(x)
    flops[0] = flops[0] + delta_flop

    #n_cell_stages= '2,3,4,3,3,1'
    n_cell_stages[0]+=1
    encodings = []
    for i in range(6):
        for j in range(n_cell_stages[i]):
            encodings.append(i)
    for idx, block in enumerate(net.blocks):
        delta_flop, x = block.get_flops(x)
        flops[encodings[idx]] += delta_flop

    delta_flop, x = net.feature_mix_layer.get_flops(x)
    flops[5] += delta_flop

    x = net.global_avg_pooling(x)
    x = x.view(x.size(0), -1)  # flatten

    delta_flop, x = net.classifier.get_flops(x)
    flops[5] += delta_flop

    return flops

if __name__ =='__main__':
    fromto = (3,4)

    width_stages = '24,32,64,96,160,320'
    n_cell_stages= '2,3,4,3,3,1'
    stride_stages= '1,2,2,1,2,1'
    conv_candidates = [
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
    ]

    width_stages =  [int(val) for val in width_stages.split(',')]
    n_cell_stages = [int(val) for val in n_cell_stages.split(',')]
    stride_stages = [int(val) for val in stride_stages.split(',')]
    
    from build_mobilenet import MobileProxylessNASNets
    teacher_net = MobileProxylessNASNets(
        width_stages=width_stages, n_cell_stages=n_cell_stages, stride_stages=stride_stages,
        n_classes= 10,
    )
    super_net = SuperProxylessNASNets(
        width_stages=width_stages, n_cell_stages=n_cell_stages, stride_stages=stride_stages,
        conv_candidates=conv_candidates, n_classes= 10, 
    )

    cifar10 = Cifar10DataProvider()

    #net = teacher_net
    net = super_net
    data_shape = [1] + list(cifar10.data_shape)
    input_var = torch.zeros(data_shape)
    with torch.no_grad():
        flops = return_flops(net, input_var, n_cell_stages)

    """
    super_net.set_chosen_op_active()
    super_net.unused_modules_off()  # remove unused module for speedup
    data_shape = [1] + list(cifar10.data_shape)
    input_var = torch.zeros(data_shape)
    with torch.no_grad():
        flop, _ = super_net.get_flops(input_var)
    """

    for idx, flop in enumerate(flops):
        print(idx,'=> Total FLOPs: %.2fM' % (flop / 1e6))
