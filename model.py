import pandas as pd
import torch
import torch.nn as nn
from GNN import user_exer_gnn, user_know_gnn, exer_user_gnn, exer_know_gnn
import torch.nn.functional as F

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class Cat_model(nn.Module):
    def __init__(self, user_num, exer_num, know_num, exp_num):
        self.user_num = user_num
        self.exer_num = exer_num
        self.know_num = know_num
        self.exp_num = exp_num
        self.prednet_input_len = self.exp_num * 6
        super(Cat_model, self).__init__()
        self.fold_i = 1
        self.emb = nn.Embedding(2, 1, max_norm=1)
        self.u_latent = nn.Embedding(self.user_num, self.exp_num)
        self.e_latent = nn.Embedding(self.exer_num, self.exp_num)
        nn.init.xavier_normal_(self.u_latent.weight)
        nn.init.xavier_normal_(self.e_latent.weight)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.know_num, self.exp_num))
        nn.init.xavier_normal_(self.knowledge_emb)
        self.stat_full = nn.Linear(self.exp_num, 1)
        self.diff_full = nn.Linear(self.exp_num, 1)
        self.disc_full = nn.Linear(self.exp_num, 1)  
        self.prednet_input_len = self.exp_num * 2 + self.know_num
        self.user_exer_gnn = user_exer_gnn(self.user_num, self.exer_num, self.know_num, self.exp_num)
        self.user_know_gnn = user_know_gnn(self.user_num, self.exer_num, self.know_num, self.exp_num)
        self.know_user_gnn = exer_user_gnn(self.user_num, self.exer_num, self.know_num, self.exp_num)
        self.know_exer_gnn = exer_know_gnn(self.user_num, self.exer_num, self.know_num, self.exp_num)
        self.user_exer, self.user_know, self.exer_know = self.data_preprocess(1)
        self.user_exer, self.user_know, self.exer_know = self.user_exer.to('cuda:0'), self.user_know.to('cuda:0'), self.exer_know.to('cuda:0')
        self.align_s = nn.Linear(self.exp_num, self.exp_num)
        self.align_e = nn.Linear(self.exp_num, self.exp_num)
        self.att_gate = nn.Linear(self.exp_num, self.exp_num)
        self.prednet_len1, self.prednet_len2 = 512, 256
        # network structure
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        self.e_guess = nn.Embedding(self.exer_num, 1)
        nn.init.constant_(self.e_guess.weight, -2.0)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, user_list, exer_list):
        self.user_exer_mat = self.user_exer_gnn(self.user_exer)
        self.user_know_mat = self.user_know_gnn(self.user_know)
        self.exer_user_mat = self.know_user_gnn(self.user_exer.transpose(0, 1))
        self.exer_know_mat = self.know_exer_gnn(self.exer_know)
        sn = self.user_exer_mat[user_list] + self.user_know_mat[user_list]
        em = self.exer_user_mat[exer_list] + self.exer_know_mat[exer_list]
        sn_a = self.align_s(sn)
        em_a = self.align_e(em)
        sim = sn_a * em_a 
        alpha = torch.sigmoid(self.att_gate(sim))
        sn = alpha * sn
        u_emb = self.u_latent(user_list)
        e_emb = self.e_latent(exer_list)
        sn = sn + u_emb
        em = em + e_emb
        batch_size = sn.size(0)
        stu_expanded = sn.unsqueeze(1).repeat(1, self.know_num, 1)
        exe_expanded = em.unsqueeze(1).repeat(1, self.know_num, 1)
        know_expanded = self.knowledge_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        stat_emb = torch.sigmoid(self.stat_full(stu_expanded * know_expanded)).squeeze(-1)
        k_difficulty = torch.sigmoid(self.diff_full(exe_expanded * know_expanded)).squeeze(-1)
        e_discrimination = F.softplus(self.disc_full(em))
        q_mask = self.exer_know[exer_list]
        AKse = e_discrimination * (stat_emb - k_difficulty) * q_mask
        input_x = torch.cat([sn, em, AKse], dim=1)
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        guess_bias = self.e_guess(exer_list)
        output = torch.sigmoid(self.prednet_full3(input_x) + guess_bias)
        return output
    def apply_clipper(self):	
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)
    def data_preprocess(self, fold_i):
        user_exer, user_know, exer_know = torch.zeros(self.user_num, self.exer_num), torch.zeros(self.user_num,
                                                                                                 self.know_num), \
                                          torch.zeros(self.exer_num, self.know_num),
        global_data = pd.read_csv(f'data/total.csv')
        for _, row in global_data.iterrows():
            exer = row['exer_id'] - 1
            log = row['knowledge_code']
            for k in eval(log):
                exer_know[exer][k - 1] = 1.0
        train_data = pd.read_csv(f'data/train_{fold_i}.csv')
        for _, row in train_data.iterrows():
            user = row['user_id'] - 1
            exer = row['exer_id'] - 1
            score = row['score']
            log = row['knowledge_code']
            user_exer[user][exer] = 1.0
            if score:
                for k in eval(log):
                    user_know[user][k - 1] = 1.0
        return user_exer, user_know, exer_know
class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
