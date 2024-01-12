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


class spa_cross_transfomer(nn.Module):
    def __init__(self, args, num_patches=49):
        super(spa_cross_transfomer, self).__init__()
        self.args = args
        self.num_patches = num_patches
        max_len = int(num_patches * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, 0.1, max_len)
        self.qk_linear = nn.Linear(self.args.trans_linear_in_dim, self.args.trans_linear_out_dim)
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim, self.args.trans_linear_out_dim)

        self.norm_qk = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.class_softmax = torch.nn.Softmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, support_set, support_labels, queries):
        # support/queries: (batch, 8, 2048, 4, 4)
        n_queries = queries.shape[0]  # 20
        n_support = support_set.shape[0]  # 25
        # (25, 8, 2048, 16)->(25, 8, 16, 2048)->(25*8, 16, 2048)
        support_set = support_set.reshape(n_support, self.args.seq_len, self.args.trans_linear_in_dim, self.num_patches).\
            permute(0, 1, 3, 2).reshape(-1, self.num_patches, self.args.trans_linear_in_dim)
        queries = queries.reshape(n_queries, self.args.seq_len, self.args.trans_linear_in_dim, self.num_patches).\
            permute(0, 1, 3, 2).reshape(-1, self.num_patches, self.args.trans_linear_in_dim)
        support_set = self.pe(support_set)
        queries = self.pe(queries)
        support_set_ks = self.qk_linear(support_set)  # 25*8 x 49 x 1152
        queries_ks = self.qk_linear(queries)  # 20*8 x 49 x 1152
        support_set_vs = self.v_linear(support_set)  # 25*8 x 49 x 1152
        queries_vs = self.v_linear(queries)  # 20*8 x 49 x 1152

        mh_support_set_ks = self.norm_qk(support_set_ks).reshape(n_support, self.args.seq_len, self.num_patches, self.args.trans_linear_out_dim)\
            .to(device)  # 25 8 x 49 x 1152
        mh_queries_ks = self.norm_qk(queries_ks).reshape(n_queries, self.args.seq_len, self.num_patches, self.args.trans_linear_out_dim)\
            .to(device)  # 20 8 x 49 x 1152
        support_labels = support_labels.to(device)
        mh_support_set_vs = support_set_vs.reshape(n_support, self.args.seq_len, self.num_patches, self.args.trans_linear_out_dim)\
            .to(device)  # 25 8 x 49 x 1152
        mh_queries_vs = queries_vs.reshape(n_queries, self.args.seq_len, self.num_patches, self.args.trans_linear_out_dim)\
            .reshape(n_queries, -1, self.args.trans_linear_out_dim).to(device)  # 20 8 x 49 x 1152

        unique_labels = torch.unique(support_labels)  # 5
        all_distances_tensor = torch.zeros(n_queries, self.args.way)  # 20 x 5

        for label_idx, c in enumerate(unique_labels):
            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(support_labels, c))  # (5,8,49,1152)
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c))  # (5,8,49,1152)
            k_bs = class_k.shape[0] # 5
            # (8, 5, 49, 1152)->(8, 5*49, 1152)
            class_k = class_k.permute(1, 0, 2, 3).reshape(self.args.seq_len, k_bs*self.num_patches, self.args.trans_linear_out_dim)
            class_v = class_v.permute(1, 0, 2, 3).reshape(self.args.seq_len, k_bs*self.num_patches, self.args.trans_linear_out_dim)

            # (20, 8, 49, 2048) (1, 8, 2048, 90)->(20, 8, 16, 90)
            class_scores = torch.matmul(mh_queries_ks, class_k.unsqueeze(0).transpose(-2, -1)) / math.sqrt(
                self.args.trans_linear_in_dim)
            class_scores = self.class_softmax(class_scores)
            class_task_support = torch.matmul(class_scores, class_v.unsqueeze(0)).reshape(n_queries, -1, self.args.trans_linear_out_dim) # 20 x 8 x 16 x 1152
            diff = mh_queries_vs - class_task_support  # 20 x 8 * 49 x 1152
            norm_sq = torch.norm(diff, dim=[-2, -1]) ** 2  # 20
            distance = torch.div(norm_sq, self.args.seq_len ** 2 * self.num_patches)
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


class CNN_Strx(nn.Module):
    def __init__(self, args):
        super(CNN_Strx, self).__init__()

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
        self.spa_trans = spa_cross_transfomer(args, num_patches=49)

    def forward(self, context_images, context_labels, target_images):
        context_features = self.resnet(context_images)  # 200 x 2048 x 7 x 7
        target_features = self.resnet(target_images)  # 160 x 2048 x 7 x 7

        context_features = context_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim, 7, 7)  # 25 x 8 x 2048 x 7 x 7
        target_features = target_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim, 7, 7)  # 20 x 8 x 2048 x 7 x 7

        # (5, 20, 8, 16, 2048)  (20, 8, 16, 2048)
        logits_frm = self.spa_trans(context_features, context_labels, target_features)['logits']

        total_logist = logits_frm

        return_dict = {'logits': split_first_dim_linear(total_logist, [NUM_SAMPLES, target_features.shape[0]])}

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
            self.trans_linear_in_dim = 2048
            self.trans_linear_out_dim = 1152

            self.way = 5
            self.shot = 1
            self.query_per_class = 1
            self.trans_dropout = 0.1
            self.seq_len = 8
            self.img_size = 224
            self.method = "resnet50"
            self.num_gpus = 1


    args = ArgsObject()
    model = CNN_Strx(args).to(device)

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
        out['logits']))