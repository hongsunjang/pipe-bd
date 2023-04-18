# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from modules.layers import *
import json


def proxyless_base(net_config=None, n_classes=1000, bn_param=(0.1, 1e-3), dropout_rate=0):
    assert net_config is not None, 'Please input a network config'
    net_config_path = download_url(net_config)
    net_config_json = json.load(open(net_config_path, 'r'))

    net_config_json['classifier']['out_features'] = n_classes
    net_config_json['classifier']['dropout_rate'] = dropout_rate

    net = ProxylessNASNets.build_from_config(net_config_json)
    net.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    return net


class MobileInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.mobile_inverted_conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)


class ProxylessNASNets(MyNetwork):

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super(ProxylessNASNets, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

    """
    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
    """

    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': ProxylessNASNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': [self.first_conv[0].config, self.first_conv[1].config],
            'blocks': [ [block.config for block in stage ] for stage in self.blocks],
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        stem = [
            set_layer_from_config(config['first_conv'][0]),
            MobileInvertedResidualBlock.build_from_config(config['first_conv'][1])
        ]
        first_conv = nn.Sequential(*stem)
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])

        blocks = []

        for stage_config in config['blocks']:
            stage = []
            for block_config in stage_config:
                stage.append(MobileInvertedResidualBlock.build_from_config(block_config))
            blocks.append(nn.Sequential(*stage))

        net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
        if ('bn' in config) and (config['bn'] is not None):
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def get_flops(self, x):
        flop = 0
        if self.first_conv is not None:
            
            if hasattr(self.first_conv, 'module'):
                first_conv = self.first_conv.module
            else:
                first_conv = self.first_conv
            for conv in first_conv:
                delta_flop, x = conv.get_flops(x)
                flop += delta_flop
    

        for stage in self.blocks:
            if hasattr(stage, 'module'):
                stage = stage.module
            for block in stage:
                delta_flop, x = block.get_flops(x)
                flop += delta_flop

        if self.feature_mix_layer is not None:
            if hasattr(self.feature_mix_layer, 'module'):
                feature_mix_layer = self.feature_mix_layer.module
            else:
                feature_mix_layer = self.feature_mix_layer

            delta_flop, x = feature_mix_layer.get_flops(x)
            flop += delta_flop

            x = self.global_avg_pooling(x)
            x = x.view(x.size(0), -1)  # flatten
            
            if hasattr(self.classifier, 'module'):
                classifier = self.classifier.module
            else:
                classifier = self.classifier
            delta_flop, x = classifier.get_flops(x)
            flop += delta_flop
        return flop, x

    def unused_stages_off(self, start, end):
        if start > 0:
            self.first_conv = None

        self.blocks = self.blocks[start:end+1] 
        
        if end < 5:
            self.feature_mix_layer = None 

    def forward(self, x, block_idx = None):

        if block_idx is not None:
            if (self.first_conv is not None) and (block_idx == 0): 
                x = self.first_conv(x)
        
            x = self.blocks[block_idx](x)
            
            if (self.feature_mix_layer is not None) and (block_idx == len(self.blocks) - 1):
                x = self.feature_mix_layer(x)
                x = self.global_avg_pooling(x)
                x = x.view(x.size(0), -1)  # flatten
                x = self.classifier(x)
        else:
            if self.first_conv is not None: 
                x = self.first_conv(x)

            for block in self.blocks:
                x = block(x)

            if self.feature_mix_layer is not None:
                x = self.feature_mix_layer(x)
                x = self.global_avg_pooling(x)
                x = x.view(x.size(0), -1)  # flatten
                x = self.classifier(x)
        return x

