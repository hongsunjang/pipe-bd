import torch
import time
from models.super_nets.super_proxyless_cifar10  import *
from data_providers.cifar10 import *

cifar10 = Cifar10DataProvider(n_worker=4)

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


def net_latency(given_net=None, device = 'cuda:0', l_type='gpu4', fast=True):
    if 'gpu' in l_type:
        l_type, batch_size = l_type[:3], int(l_type[3:])
    else:
        batch_size = 1

    data_shape = [batch_size] + list(cifar10.data_shape)

    if given_net is not None:
        net = given_net

    if l_type == 'cpu':
        if fast:
            n_warmup = 1
            n_sample = 2
        else:
            n_warmup = 10
            n_sample = 100
        """
        try:
            net.net_on_cpu_for_latency.set_active_via_net(net)
        except AttributeError:
            print(type(self.net_on_cpu_for_latency), ' do not `support set_active_via_net()`')
        net = self.net_on_cpu_for_latency
        """
        net = net.cpu()
        images = torch.zeros(data_shape, device=torch.device('cpu'))
    elif l_type == 'gpu':
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 100
        images = torch.zeros(data_shape, device=device)
    else:
        raise NotImplementedError

    measured_latency = {'warmup': [], 'sample': [], 'stage':np.zeros(len(n_cell_stages))}
    
    n_cell_stages[0]+=1
    encodings = []
    for i in range(6):
        for j in range(n_cell_stages[i]):
            encodings.append(i)
    net.train()
    for i in range(n_warmup + n_sample):
        # set random path
        #net.set_chosen_op_active()
        net.reset_binary_gates()
        # remove unused modules
        net.unused_modules_off()
        start_time = time.time()
        l_values = np.zeros(len(n_cell_stages))
        x = net.first_conv(images)

        delta_time = time.time() - start_time
        l_values[0] += delta_time
        delta_time = time.time()

        for idx, block in enumerate(net.blocks):
            x = block(x)
            delta_time = time.time() - delta_time
            l_values[encodings[idx]] += delta_time
            delta_time = time.time()

        x = net.feature_mix_layer(x)
        delta_time = time.time() - delta_time
        l_values[5] += delta_time
        delta_time = time.time()

        x = net.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten

        x = net.classifier(x)
        delta_time = time.time() - delta_time
        l_values[5] += delta_time

        #net(images)
        net.unused_modules_back()
        used_time = (time.time() - start_time) * 1e3  # ms
        if i >= n_warmup:
            measured_latency['stage'] += (l_values * 1e3)
            measured_latency['sample'].append(used_time)
        else:
            measured_latency['warmup'].append(used_time)
    return sum(measured_latency['sample']) / n_sample, measured_latency['stage'] /n_sample

if __name__ =='__main__':
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

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = super_net.to(device)
    net.init_arch_params()
    l_avg, l_values = net_latency(net, device, l_type='gpu256', fast=False)
    print("Avg:",l_avg)
    print("Per stage:", l_values )
