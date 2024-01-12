import torch
import torch.nn as nn
from collections import OrderedDict
from utils import split_first_dim_linear
import math
import numpy as np
from itertools import combinations
from einops import rearrange
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import random
NUM_SAMPLES = 1
np.random.seed(3483)
torch.manual_seed(3483)
torch.cuda.manual_seed(3483)
torch.cuda.manual_seed_all(3483)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """ Implement the PE function. """
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        # pe is of shape max_len(5000) x 2048(last layer of FC)
        pe = torch.zeros(max_len, d_model)
        # position is of shape 5000 x 1
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        # pe contains a vector of shape 1 x 5000 x 2048
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class TemporalCrossTransformer(nn.Module):
    """ Original TRX """
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()

        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size,
                                  self.args.trans_linear_out_dim)  # .cuda()
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size,
                                  self.args.trans_linear_out_dim)  # .cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)

        self.class_softmax = torch.nn.Softmax(dim=1)

        # generate all ordered tuples corresponding to the temporal set size 2 or 3.
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples_len = len(self.tuples)  # 28

    def forward(self, support_set, support_labels, queries):
        # support_set : 25 x 8 x 2048, support_labels: 25, queries: 20 x 8 x 2048
        n_queries = queries.shape[0]  # 20
        n_support = support_set.shape[0]  # 25

        # static pe after adding the position embedding
        support_set = self.pe(support_set)  # Support set is of shape 25 x 8 x 2048 -> 25 x 8 x 2048
        queries = self.pe(queries)  # Queries is of shape 20 x 8 x 2048 -> 20 x 8 x 2048

        # construct new queries and support set made of tuples of images after pe
        # Support set s = number of tuples(28 for 2/56 for 3) stacked in a list form containing elements of form 25 x 4096(2 x 2048 - (2 frames stacked))
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]

        support_set = torch.stack(s, dim=-2)  # 25 x 28 x 4096
        queries = torch.stack(q, dim=-2)  # 20 x 28 x 4096

        # apply linear maps for performing self-normalization in the next step and the key map's output
        '''
            support_set_ks is of shape 25 x 28 x 1152, where 1152 is the dimension of the key = query head. converting the 5-way*5-shot x 28(tuples).
            query_set_ks is of shape 20 x 28 x 1152 covering 4 query/sample*5-way x 28(number of tuples)
        '''
        support_set_ks = self.k_linear(support_set)  # 25 x 28 x 1152
        queries_ks = self.k_linear(queries)  # 20 x 28 x 1152
        support_set_vs = self.v_linear(support_set)  # 25 x 28 x 1152
        queries_vs = self.v_linear(queries)  # 20 x 28 x 1152

        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks).to(device)  # 25 x 28 x 1152
        mh_queries_ks = self.norm_k(queries_ks).to(device)  # 20 x 28 x 1152
        support_labels = support_labels.to(device)
        mh_support_set_vs = support_set_vs.to(device)  # 25 x 28 x 1152
        mh_queries_vs = queries_vs.to(device)  # 20 x 28 x 1152

        unique_labels = torch.unique(support_labels)  # 5

        # init tensor to hold distances between every support tuple and every target tuple. It is of shape 20  x 5
        '''
            4-queries * 5 classes x 5(5 classes) and store this in a logit vector
        '''
        all_distances_tensor = torch.zeros(n_queries, self.args.way)  # 20 x 5

        for label_idx, c in enumerate(unique_labels):
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0,
                                         self._extract_class_indices(support_labels, c))  # 5 x 28 x 1152
            class_v = torch.index_select(mh_support_set_vs, 0,
                                         self._extract_class_indices(support_labels, c))  # 5 x 28 x 1152
            k_bs = class_k.shape[0]  # 5

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2, -1)) / math.sqrt(
                self.args.trans_linear_out_dim)  # 20 x 5 x 28 x 28

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0, 2, 1, 3)  # 20 x 28 x 5 x 28

            # [For the 20 queries' 28 tuple pairs, find the best match against the 5 selected support samples from the same class
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)  # 20 x 28 x 140
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]  # list(20) x 28 x 140
            class_scores = torch.cat(class_scores)  # 560 x 140 - concatenate all the scores for the tuples
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)  # 20 x 28 x 5 x 28
            class_scores = class_scores.permute(0, 2, 1, 3)  # 20 x 5 x 28 x 28

            # get query specific class prototype
            query_prototype = torch.matmul(class_scores, class_v)  # 20 x 5 x 28 x 1152
            query_prototype = torch.sum(query_prototype, dim=1).to(
                device)  # 20 x 28 x 1152 -> Sum across all the support set values of the corres. class

            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype  # 20 x 28 x 1152
            norm_sq = torch.norm(diff, dim=[-2, -1]) ** 2  # 20
            distance = torch.div(norm_sq, self.tuples_len)  # 20

            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:, c_idx] = distance  # 20

        return_dict = {'logits': all_distances_tensor}

        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class mSEModule(nn.Module):
    """ TDSAM """
    def __init__(self, channel, n_segment=8):
        super(mSEModule, self).__init__()
        self.channel = channel
        self.reduction = 2
        self.n_segment = n_segment
        self.temperature = self.channel // self.reduction
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv = nn.Conv3d(in_channels=self.channel,
                              out_channels=self.channel,
                              kernel_size=(3, 1, 1),
                              stride=(1, 1, 1),
                              padding=tuple(x // 2 for x in (3, 1, 1)), groups=self.channel, bias=False)
        self.conv1 = nn.Conv2d(in_channels=self.channel,
                               out_channels=self.channel // self.reduction,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel // self.reduction)
        self.conv2 = nn.Conv2d(in_channels=self.channel // self.reduction,
                               out_channels=self.channel // self.reduction,
                               kernel_size=3, padding=1, groups=self.channel // self.reduction, bias=False)

        self.avg_pool_forward2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_backward2 = nn.AvgPool2d(kernel_size=2, stride=2)  # nn.AdaptiveMaxPool2d(1)

        self.pad1_forward = (0, 0, 0, 0, 0, 0, 0, 1)
        self.pad1_backward = (0, 0, 0, 0, 0, 0, 1, 0)

        self.conv3 = nn.Conv2d(in_channels=self.channel // self.reduction,
                               out_channels=self.channel // self.reduction, kernel_size=1, bias=False)

        self.conv3_smallscale2 = nn.Conv2d(in_channels=self.channel // self.reduction,
                                           out_channels=self.channel // self.reduction, padding=1, kernel_size=3,
                                           bias=False)
        self.bn3_smallscale2 = nn.BatchNorm2d(num_features=self.channel // self.reduction)

        self.conv3_smallscale4 = nn.Conv2d(in_channels=self.channel // self.reduction,
                                           out_channels=self.channel // self.reduction, padding=1, kernel_size=3,
                                           bias=False)
        self.bn3_smallscale4 = nn.BatchNorm2d(num_features=self.channel // self.reduction)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        bottleneck = self.conv1(x)  # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck)  # nt, c//r, h, w
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w

        t_fea_forward, _ = reshape_bottleneck.split([self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w
        _, t_fea_backward = reshape_bottleneck.split([1, self.n_segment - 1], dim=1)  # n, t-1, c//r, h, w
        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view(
            (-1, self.n_segment) + conv_bottleneck.size()[1:])  # n, t, c//r, h, w
        _, tPlusone_fea_forward = reshape_conv_bottleneck.split([1, self.n_segment - 1], dim=1)  # n, t-1, c//r, h, w
        tPlusone_fea_backward, _ = reshape_conv_bottleneck.split([self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w
        diff_fea_forward = tPlusone_fea_forward - t_fea_forward  # n, t-1, c//r, h, w
        diff_fea_backward = tPlusone_fea_backward - t_fea_backward  # n, t-1, c//r, h, w
        diff_fea_pluszero_forward = F.pad(diff_fea_forward, self.pad1_forward, mode="constant",
                                          value=0)  # n, t, c//r, h, w
        diff_fea_pluszero_forward = diff_fea_pluszero_forward.view(
            (-1,) + diff_fea_pluszero_forward.size()[2:])  # nt, c//r, h, w
        diff_fea_pluszero_backward = F.pad(diff_fea_backward, self.pad1_backward, mode="constant",
                                           value=0)  # n, t, c//r, h, w
        diff_fea_pluszero_backward = diff_fea_pluszero_backward.view(
            (-1,) + diff_fea_pluszero_backward.size()[2:])  # nt, c//r, h, w
        y_forward_smallscale2 = self.avg_pool_forward2(diff_fea_pluszero_forward)  # nt, c//r, h//2, w//2
        y_backward_smallscale2 = self.avg_pool_backward2(diff_fea_pluszero_backward)  # nt, c//r, h//2, w//2

        y_forward_smallscale4 = diff_fea_pluszero_forward
        y_backward_smallscale4 = diff_fea_pluszero_backward
        y_forward_smallscale2 = self.bn3_smallscale2(self.conv3_smallscale2(y_forward_smallscale2))
        y_backward_smallscale2 = self.bn3_smallscale2(self.conv3_smallscale2(y_backward_smallscale2))

        y_forward_smallscale4 = self.bn3_smallscale4(self.conv3_smallscale4(y_forward_smallscale4))
        y_backward_smallscale4 = self.bn3_smallscale4(self.conv3_smallscale4(y_backward_smallscale4))

        y_forward_smallscale2 = F.interpolate(y_forward_smallscale2, diff_fea_pluszero_forward.size()[2:])
        y_backward_smallscale2 = F.interpolate(y_backward_smallscale2, diff_fea_pluszero_backward.size()[2:])
        # y_f:(40, 128, 7, 7), y_b:(40, 128, 7, 7)
        y_forward = self.conv3(
            1.0 / 3.0 * diff_fea_pluszero_forward + 1.0 / 3.0 * y_forward_smallscale2 + 1.0 / 3.0 * y_forward_smallscale4)  # nt, c, 1, 1
        y_backward = self.conv3(
            1.0 / 3.0 * diff_fea_pluszero_backward + 1.0 / 3.0 * y_backward_smallscale2 + 1.0 / 3.0 * y_backward_smallscale4)  # nt, c, 1, 1
        y_forward = y_forward.reshape(y_forward.size()[:-2] + (-1,)).permute(0, 2, 1)
        y_backward = y_backward.reshape(y_backward.size()[:-2] + (-1,)).permute(0, 2, 1)

        attn1 = torch.bmm(y_forward, y_forward.transpose(1, 2))
        attn2 = torch.bmm(y_backward, y_backward.transpose(1, 2))
        attn1 = attn1 / np.power(self.temperature, 0.5)
        attn2 = attn2 / np.power(self.temperature, 0.5)
        attn1 = self.softmax(attn1)
        attn2 = self.softmax(attn2)
        attn1 = self.dropout(attn1)
        attn2 = self.dropout(attn2)
        v = x.reshape((-1, self.n_segment) + x.size()[1:]).permute(0, 2, 1, 3, 4)
        v = self.conv(v)
        v = v.permute(0, 2, 1, 3, 4).reshape(-1, self.channel, 7, 7)
        v = v.reshape(v.size()[:-2] + (-1,)).permute(0, 2, 1)
        activate_v1 = torch.bmm(attn1, v)
        activate_v2 = torch.bmm(attn2, v)
        activate = 0.5 * activate_v1 + 0.5 * activate_v2
        activate = activate.permute(0, 2, 1).reshape(-1, self.channel, 7, 7)
        output = x + activate * self.gamma
        return output

class DistanceLoss(nn.Module):
    " CTRX: Compute the Query-class similarity on the patch-enriched features. "
    def __init__(self, args, temporal_set_size=2):
        super(DistanceLoss, self).__init__()
        self.args = args
        self.temporal_set_size = temporal_set_size
        max_len = int(self.args.seq_len * 1.5)
        self.dropout = nn.Dropout(p=0.1)
        self.change_dim = self.args.trans_linear_in_dim
        # generate all ordered tuples corresponding to the temporal set size 2 or 3.
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples_len = len(self.tuples)  # 28 for tempset_2

        # nn.Linear(4096, 1024)
        self.clsW = nn.Linear(self.args.trans_linear_in_dim * self.temporal_set_size, self.args.trans_linear_out_dim).to(device)
        self.relu = torch.nn.ReLU(inplace=False).to(device)
        
    def forward(self, support_set, support_labels, queries):
        # support_set : 25 x 8 x 2048, support_labels: 25, queries: 20 x 8 x 2048
        n_queries = queries.shape[0] #20
        n_support = support_set.shape[0] #25     

        # Add a dropout before creating tuples
        support_set = self.dropout(support_set) # 25 x 8 x 2048
        queries = self.dropout(queries) # 20 x 8 x 2048

        # construct new queries and support set made of tuples of images after pe
        # Support set s = number of tuples(28 for 2/56 for 3) stacked in a list form containing elements of form 25 x 4096(2 x 2048 - (2 frames stacked))
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]

        support_set = torch.stack(s, dim=-2) # 25 x 28 x 4096
        queries = torch.stack(q, dim=-2) # 20 x 28 x 4096

        support_labels = support_labels
        unique_labels = torch.unique(support_labels).to(device) # 5

        query_embed = self.clsW(queries.view(-1, self.change_dim *self.temporal_set_size)) # 560[20x28] x 2048*2

        # Add relu after clsW
        query_embed = self.relu(query_embed) # 560 x 1024 

        # init tensor to hold distances between every support tuple and every target tuple. It is of shape 20  x 5
        '''
            4-queries * 5 classes x 5(5 classes) and store this in a logit vector
        '''
        dist_min = torch.zeros(n_queries, self.args.way).to(device) # 20 x 5
        dist_max = torch.zeros(n_queries, self.args.way).to(device) # 20 x 5

        support_all = torch.zeros(self.args.way, self.args.shot * self.tuples_len,  self.args.trans_linear_out_dim).to(device) # 5 x 140 x 4096
        pos_all = torch.zeros(self.args.way, n_queries * self.tuples_len).to(device) # 5 x 560
        ave_dist_all = torch.zeros(self.args.way, n_queries * self.tuples_len).to(device)  # 5 x 560

        for label_idx, c in enumerate(unique_labels):
            c_idx = c.long()  

            class_k = torch.index_select(support_set, 0, self._extract_class_indices(support_labels, c)) # 5 x 28 x 4096
            class_k = class_k.view(-1,  self.change_dim*self.temporal_set_size) # 140 x 4096
            
            support_embed = self.clsW(class_k)  # 140[5 x 28] x 4096
            support_embed = self.relu(support_embed) # 140 x 1024  

            # 所有support特征
            support_all[c_idx, :, :] = support_embed  

            # Calculate p-norm distance between the query embedding and the support set embedding
            # distmat = torch.cosine_similarity(query_embed.unsqueeze(1), support_embed.unsqueeze(0),dim=-1) # 560[20 x 28] x 140[28 x 5]
            distmat = torch.cdist(query_embed, support_embed) # 560[20 x 28] x 140[28 x 5]
            
            # Across the 140 tuples compared against
            dist_class = distmat.view(-1, self.tuples_len, self.args.shot) # 560 x 28 x 5
            dist_class = dist_class.topk(dim=1, largest=True, k=1)[0].mean(dim=1) # 560 x 3 x 5 - > 560 x 5
            ave_dist = dist_class.mean(dim=1)  # 560

            ave_dist_all[c_idx,:] = ave_dist 

            # Across the 140 tuples compared against, get the minimum and maxnum distance for each of the 560 queries
            min_dist = distmat.min(dim=1)[0].reshape(n_queries, self.tuples_len) # 20[5-way x 4-queries] x 28
            max_dist = distmat.max(dim=1)[0].reshape(n_queries, self.tuples_len) # 20[5-way x 4-queries] x 28

            max_pos = distmat.argmax(dim=1)  # 560
            pos_all[c_idx,:] = max_pos

            # Average across the 28 tuples
            query_dist_min = min_dist.mean(dim=1)  # 20
            query_dist_max = max_dist.mean(dim=1)  # 20

            # Make it negative as this has to be reduced.
            min_distance = 1.0 * query_dist_min
            dist_min[:,c_idx] = min_distance # Insert into the required location.

            max_distance = query_dist_max
            dist_max[:,c_idx] = max_distance # Insert into the required location.

        """ """
        dist_contrast = torch.zeros(n_queries, self.args.way).to(query_embed.device) # 20 x 5

        for label_idx, c_1 in enumerate(unique_labels):

            c_idx_1 = c_1.long() 
            support_embed = support_all[c_idx_1]  # 140 x 4096

            min_support_embed = torch.index_select(support_embed,dim=0,index=pos_all[c_idx_1].int()) # 560 x 1024 ---- c_1类别的support set与的query set中视频最相关的片段的下标
            
            other_index = unique_labels[~np.isin(unique_labels.cpu(), c_1.cpu())]
            support_other_embed = torch.index_select(support_all,dim=0,index=other_index.int()).view(-1,  self.args.trans_linear_out_dim) # 4 x 140 or 28 x 1024 ->560 or 112 x 1024

            # cos_distmat = torch.cosine_similarity(min_support_embed.unsqueeze(dim=1), support_other_embed.unsqueeze(0),dim=-1)  # 560 x 560(4x140)
            cos_distmat = torch.cdist(min_support_embed, support_other_embed)  # 560 x 560(4x140)

            support_other_embed = support_other_embed.view(self.args.way-1, self.args.shot * self.tuples_len, -1)  # 4 x (140 or 28) x 1024 
            cos_distmat = cos_distmat.view(-1, self.args.way-1, self.args.shot * self.tuples_len).permute(1,0,2)  # 560 x 4 x (140 or 28) -> 4 x 560 x (140 or 28)

            record_pos = torch.zeros(self.args.way-1, self.args.shot * self.tuples_len)  # 4 x (140 or 28)

            # 转list加速
            cos_distmat_np = cos_distmat.cpu().detach().numpy().tolist()
            record_pos_np = record_pos.cpu().detach().numpy().tolist()
            ave_dist_all_np = ave_dist_all.cpu().detach().numpy().tolist()

            all_index = []
            kind = c_idx_1.cpu()
            for idx in range(4):
                for s1 in range(cos_distmat.shape[1]):
                    for s2 in range(cos_distmat.shape[2]):
                        if cos_distmat_np[idx][s1][s2] > ave_dist_all_np[kind][s1]:
                            record_pos_np[idx][s2] = record_pos_np[idx][s2] + 1
                            
                record_pos = torch.from_numpy(np.array(record_pos_np)).to(device)
                
                nonzero_num = torch.count_nonzero(record_pos[idx]).item()
                nonzero_index = torch.nonzero(record_pos[idx]<record_pos[idx].sum().div(nonzero_num)).view(-1).to(query_embed.device)  # 获取不相关片段的下表    
                all_index.append(nonzero_index)
                    
            # print(all_index[0])

            diff_embed=[torch.index_select(support_other_embed[m], 0, all_index[m].int()).to(query_embed.device) for m in range(self.args.way-1)] # m x 1024
            diff_embed = torch.cat(diff_embed, dim=0)
            
            # distmat_diff = torch.cosine_similarity(query_embed.unsqueeze(dim=1), diff_embed.unsqueeze(0),dim=-1) # 560 x m
            distmat_diff = torch.cdist(query_embed, diff_embed)  # 560 x m

            distmat_diff = distmat_diff.mean(dim=1).reshape(n_queries, self.tuples_len) # 20[5-way x 4-queries] x 28
            # Average across the 28 tuples
            query_dist_diff = distmat_diff.mean(dim=1).div(self.args.way-1)  # 20
            dist_contrast[:, c_idx_1] = dist_contrast[:, c_idx_1] + query_dist_diff
        
        dist_sum = dist_contrast + dist_max
        dist_contrast_all = torch.div(dist_max, dist_sum)
        # print(dist_contrast_all)
        return_dict = {'logits_max': dist_max, 'logits_contrast': dist_contrast_all}
        
        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector
        


class CNN_BiMACL(nn.Module):
    """
        Standard Video Backbone connected to a Temporal Cross Transformer, Query Distance
        Similarity Loss and Patch-level and Frame-level Attention Blocks.
    """

    def __init__(self, args):
        super(CNN_BiMACL, self).__init__()

        self.train()
        self.args = args

        # Using ResNet Backbone
        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)

        last_layer_idx = -2
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        self.Motion_Excitation = mSEModule(self.args.trans_linear_in_dim, n_segment=self.args.seq_len)
        # Temporal Cross Transformer for modelling temporal relations
        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set])
        # New-distance metric for post patch-level enriched features
        self.new_dist_loss_post_pat = [DistanceLoss(args, s) for s in self.args.temp_set]

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, context_images, context_labels, target_images):
        '''
            context_features/target_features is of shape (num_images x 2048) [final Resnet FC layer] after squeezing
        '''
        '''
            context_images: 200 x 3 x 224 x 224, target_images = 160 x 3 x 224 x 224
        '''
        context_features = self.resnet(context_images)  # 200 x 2048 x 7 x 7
        target_features = self.resnet(target_images)  # 160 x 2048 x 7 x 7

        context_features = self.Motion_Excitation(context_features)
        target_features = self.Motion_Excitation(target_features)

        # (200, 2048)->(25, 8, 2048) and (160, 2048)->(20, 8, 2048)
        context_features = self.avg_pool(context_features).squeeze().reshape(-1, self.args.seq_len,
                                                                             self.args.trans_linear_in_dim)
        target_features = self.avg_pool(target_features).squeeze().reshape(-1, self.args.seq_len,
                                                                           self.args.trans_linear_in_dim)

        all_logits_fr = [t(context_features, context_labels, target_features)['logits'] for t in self.transformers]
        all_logits_fr = torch.stack(all_logits_fr, dim=-1)
        sample_logits_fr = all_logits_fr
        sample_logits_fr = torch.mean(sample_logits_fr, dim=[-1])  # 20 x 5

        # TRX logistics
        # Compute logits using the new loss before applying frame-level attention
        all_logits_post_pat = [n(context_features, context_labels, target_features) for n in
                               self.new_dist_loss_post_pat]
        all_logits_logits_contrast = [x["logits_contrast"] for x in all_logits_post_pat]
        all_logits_logits_max = [x["logits_max"] for x in all_logits_post_pat]
        all_logits_logits_contrast = torch.stack(all_logits_logits_contrast, dim=-1)
        all_logits_logits_contrast = torch.mean(all_logits_logits_contrast, dim=[-1])
        all_logits_logits_max = torch.stack(all_logits_logits_max, dim=-1)
        all_logits_logits_max = torch.mean(all_logits_logits_max, dim=[-1])

        return_dict = {'logits': split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]]),
                       'all_logits_logits_contrast': split_first_dim_linear(all_logits_logits_contrast, [NUM_SAMPLES, target_features.shape[0]]),
                       'all_logits_logits_max': split_first_dim_linear(all_logits_logits_max, [NUM_SAMPLES, target_features.shape[0]])}

        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])


if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 512

            self.way = 5
            self.shot = 1
            self.query_per_class = 2
            self.trans_dropout = 0.1
            self.seq_len = 8
            self.img_size = 224
            self.method = "resnet18"
            self.num_gpus = 1
            self.temp_set = [2, 3]


    args = ArgsObject()
    model = CNN_BiMACL(args).to(device)

    support_imgs = torch.rand(args.way * args.shot * args.seq_len, 3, args.img_size, args.img_size).to(device)
    target_imgs = torch.rand(args.way * args.query_per_class * args.seq_len, 3, args.img_size, args.img_size).to(device)
    support_labels = torch.tensor([0, 1, 2, 3, 4]).to(device)
    from thop import profile
    import time

    flops, params = profile(model, inputs=(support_imgs, support_labels, target_imgs))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
    begin = time.time()
    out = model(support_imgs, support_labels, target_imgs)
    end = time.time()
    print("time is:", end - begin)

    print("M2ACL returns the distances from each query to each class prototype.  Use these as logits.  Shape: {}".format(
        out['logits'].shape))



