import torch
from .pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample

def farthest_pts_sampling_tensor(pts, num_samples, return_sampled_idx=False):
    '''
    :param pts: bn, n, 3
    :param num_samples:
    :return:
    '''
    
    sampled_pts_idx = furthest_point_sample(pts[:,:,:3].contiguous(), num_samples)
    sampled_pts_idx_viewed = sampled_pts_idx.view(sampled_pts_idx.shape[0]*sampled_pts_idx.shape[1]).cuda().type(torch.LongTensor)
    batch_idxs = torch.tensor(range(pts.shape[0])).type(torch.LongTensor)
    batch_idxs_viewed = batch_idxs[:, None].repeat(1, sampled_pts_idx.shape[1]).view(batch_idxs.shape[0]*sampled_pts_idx.shape[1])
    sampled_pts = pts[batch_idxs_viewed, sampled_pts_idx_viewed, :]
    sampled_pts = sampled_pts.view(pts.shape[0], num_samples, 9)

    if return_sampled_idx == False:
        return sampled_pts
    else:
        return sampled_pts, sampled_pts_idx