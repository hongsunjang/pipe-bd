import argparse
import time

from models.super_nets.super_proxyless import *

class MobileProxylessNASNets(ProxylessNASNets):
    def __init__(self, width_stages, n_cell_stages, stride_stages, n_classes=10,  bn_param=(0.1, 1e-3), dropout_rate=0):
        first_k = 3
        if n_classes == 1000:
            first_s = 2
        else :
            first_s = 1

        input_channel = make_divisible(32, 8)
        first_cell_width = make_divisible(16, 8)
        first_conv = ConvLayer(
            3, input_channel, kernel_size=first_k, stride=first_s, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
        )
        first_block = MobileInvertedResidualBlock(
                MBInvertedConvLayer(input_channel, first_cell_width, kernel_size= 3, stride= 1, expand_ratio= 1),
                    None
                )
        stem = nn.Sequential(first_conv, first_block)

        blocks= []

        input_channel = first_cell_width
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            stage = []

            nb = n_cell
            output_channel = width
            
            b = MBInvertedConvLayer(input_channel, output_channel, kernel_size= 3, stride= s, expand_ratio= 6)
            stage.append(MobileInvertedResidualBlock(b, None))

            for _ in range(nb-1): # Normal blocks
                b = MBInvertedConvLayer(output_channel, output_channel, kernel_size= 3, stride=1, expand_ratio= 6)
                shortcut = IdentityLayer(input_channel, input_channel) 
                stage.append(MobileInvertedResidualBlock(b, shortcut))
            
            blocks.append(nn.Sequential(*stage))
            input_channel = output_channel

        last_channel = 1280
        feature_mix_layer = ConvLayer(
            input_channel, last_channel, kernel_size=1, stride = 1, use_bn=True, act_func='relu6', ops_order='weight_bn_act',
        )

        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        super(MobileProxylessNASNets, self).__init__(stem, blocks, feature_mix_layer, classifier)
        
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])



