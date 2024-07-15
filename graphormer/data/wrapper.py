# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import torch_geometric
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from functools import lru_cache
import pyximport
import torch.distributed as dist

pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos

'''
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    在默认 offset=512 的情况下：feature_num = 3  # 因为 x.size(1) 是 3
    生成的 feature_offset 为 tensor([  1, 513, 1025])
    结果为 tensor([[   2,  515, 1028], [   5,  518, 1031]])
    这个函数的作用是为输入特征添加唯一的偏移量,以便后续的嵌入操作能够区分不同特征。这样可以避免特征值的冲突,使得每个特征都在不同的范围内,从而保证嵌入索引的唯一性。
    BYD
    我估摸着是因为输入特征普遍在1-64,要不就是one hot编码,所以说加一加就不是one hot了。
    我懂了,就是纯纯的把一堆one-hot编码和小编号混合起来,像查表一样,比如说010,对应0+512+1+512*2+0=1537;
    如果还是010,对应仍然是1537,如果是410,对应的就是4+512+1+512*2+0=1541,相当于一个512进制。
    确定一个唯一标识之后,就像查表一样,查出来1541对应的embedding向量。
'''
@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    # x是输入特征,要把010和100区分开
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True 

    # edge feature here
    if len(edge_attr.size()) == 1:
        # 如果edge_attr只有一维
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        # edge_attr是模型默认输入,我们全是1
        convert_to_single_emb(edge_attr) + 1
    )

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()

    return item


class MyPygPCQM4MDataset(PygPCQM4Mv2Dataset):
    def download(self):
        super(MyPygPCQM4MDataset, self).download()

    def process(self):
        super(MyPygPCQM4MDataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        return preprocess_item(item)


class MyPygGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        if dist.get_rank() == 0:
            super(MyPygGraphPropPredDataset, self).download()
        dist.barrier()

    def process(self):
        if dist.get_rank() == 0:
            super(MyPygGraphPropPredDataset, self).process()
        dist.barrier()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        item.y = item.y.reshape(-1)
        return preprocess_item(item)
