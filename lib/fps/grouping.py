'''
PointGroup operations
Written by Li Jiang
'''

import torch
from torch.autograd import Function

import pclib


class Grouping(Function):
    @staticmethod
    def forward(ctx, feats, map_rule, average=True):
        '''
        :param ctx:
        :param map_rule: cuda int M * (maxActive + 1)
        :param feats: cuda float N * C
        :return: output_feats: cuda float M * C
        '''
        assert map_rule.is_contiguous()
        assert feats.is_contiguous()

        N, C = feats.size()

        assert N == map_rule.size(0)

        M = map_rule.max() + 1
        output_feats = torch.cuda.FloatTensor(M, C).zero_()

        pclib.grouping_fp(feats, output_feats, map_rule, N, C)

        if not average:
            ctx.for_backwards = (map_rule, N, None)
            return output_feats
        else:
            counts = torch.cuda.FloatTensor(M, 1).zero_()
            index, ele_count = torch.unique(map_rule, return_counts=True)
            counts[index, 0] = ele_count.float()

            ctx.for_backwards = (map_rule, N, counts)
            return output_feats / counts


    @staticmethod
    def backward(ctx, d_output_feats):
        map_rule, N, counts = ctx.for_backwards
        M, C = d_output_feats.size()

        if counts is None:
            ct_d_output_feats = d_output_feats.contiguous()
        else:
            ct_d_output_feats = d_output_feats.contiguous() / counts

        d_feats = torch.cuda.FloatTensor(N, C).zero_()
        pclib.grouping_bp(ct_d_output_feats, d_feats, map_rule, N, C)

        return d_feats, None, None


grouping = Grouping.apply

