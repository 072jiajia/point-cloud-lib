'''
PointGroup operations
Written by Li Jiang
'''

import torch
from torch.autograd import Function

import pclib


class AvgGrouping(Function):
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

        out_feats = pclib.avg_grouping_fp(feats, map_rule, N, C, M)

        if not average:
            ctx.for_backwards = (map_rule, N, None)
            return out_feats
        else:
            counts = pclib.count_elements(map_rule, M)
            ctx.for_backwards = (map_rule, N, counts)
            return out_feats / counts


    @staticmethod
    def backward(ctx, inp_grad):
        map_rule, N, counts = ctx.for_backwards
        M, C = inp_grad.size()

        if counts is None:
            ct_inp_grad = inp_grad.contiguous()
        else:
            ct_inp_grad = inp_grad.contiguous() / counts

        out_grad = pclib.avg_grouping_bp(ct_inp_grad, map_rule, N, C)

        return out_grad, None, None



class MaxGrouping(Function):
    @staticmethod
    def forward(ctx, feats, map_rule):
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

        max_index, max_value = pclib.max_grouping_fp(feats, map_rule, M)

        ctx.for_backwards = (max_index, N)
        return max_value


    @staticmethod
    def backward(ctx, inp_grad):
        max_index, N = ctx.for_backwards
        M, C = inp_grad.size()

        ct_inp_grad = inp_grad.contiguous()

        out_grad = pclib.max_grouping_bp(ct_inp_grad, max_index, N)

        return out_grad, None




avg_grouping = AvgGrouping.apply
max_grouping = MaxGrouping.apply

