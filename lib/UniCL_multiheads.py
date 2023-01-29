import torch
import torch.nn as nn
import copy
import os
from timm.models.layers import trunc_normal_
from collections import OrderedDict

#For FLR1.5 models, switch to bixi/v1.5.0 branch
from Florence.UniCLModel import load_opt_from_config_file, UniCLModel

class UniCL_multiheads(nn.Module):
    def init_heads(self, init_type='bias_zero', num_classes_list = None, heads_labelmap = None):
        if num_classes_list is not None:
            self.heads_num_classes = num_classes_list

        if heads_labelmap is not None:
            self.heads_labelmap = heads_labelmap

        for head_index in range(self.num_heads):
            if num_classes_list is not None:
                self.heads_list[head_index][0] = nn.Linear(self.embed_dim, num_classes_list[head_index])

            m = self.heads_list[head_index][0]

            if init_type == 'bias_zero':
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)
            elif init_type == 'bias_prior_prob':
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, bias_value)
            else:
                raise NotImplementedError

    def __init__(self, cfg, num_classes_list, heads_labelmap, model_config_file='', activation=None, head_index=None):
        super(UniCL_multiheads, self).__init__()

        assert (len(num_classes_list) > 0)
        self.num_heads = len(num_classes_list)
        self.heads_num_classes = num_classes_list

        config = load_opt_from_config_file(model_config_file)
        config['IMAGE_ENCODER']['SPEC']['DROP_PATH_RATE'] = cfg.MODEL.DROP_PATH
        config['VERBOSE'] = cfg.MODEL.VERBOSE
        model = UniCLModel(config)

        if config['UNICL_MODEL']['LOAD_PRETRAINED']:
            model.from_pretrained(
                config['UNICL_MODEL']['PRETRAINED'],
                config['UNICL_MODEL']['PRETRAINED_LAYERS'],
                config['VERBOSE']
            )

        self.feature_common = copy.deepcopy(model.image_encoder)
        del model

        self.embed_dim = self.feature_common.embed_dims[-1]

        self.heads_list = nn.ModuleList()
        self.head_index = head_index

        for head_index in range(self.num_heads):
            assert (num_classes_list[head_index] > 0)
            head = OrderedDict()
            m = nn.Linear(self.embed_dim, num_classes_list[head_index])

            head["head_{}".format(head_index)] = m

            if activation == 'sigmoid':
                head["head_{}_{}".format(head_index, activation)] = nn.Sigmoid()

            self.heads_list.append(nn.Sequential(head))

        self.init_heads(init_type=cfg.MODEL.CLASSIFIER_INIT_TYPE)

        self.heads_labelmap = heads_labelmap

    def forward(self, x, head_index=None):
        x = self.feature_common.forward_features(x)
        if head_index is None:
            return [self.heads_list[idx](x) for idx in range(self.num_heads)]
        output = self.heads_list[head_index](x)

        return output