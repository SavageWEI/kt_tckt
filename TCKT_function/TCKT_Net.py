
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7" # 0, 1, 2, ...


import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
py_name = os.path.basename(__file__).split(".")[0]
print("net_test_best_global_dict")

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def _init_fn(worker_id):
    np.random.seed(int(seed))

class TCKTNet(nn.Module):
    def __init__(self, n_at, n_it, n_exercise, n_concept, d_a, d_e, d_k, q_matrix, dropout=0.2):

        super(TCKTNet, self).__init__()
        self.d_k = d_k
        self.d_a = d_a
        self.d_e = d_e
        self.q_matrix = q_matrix
        self.n_concept = n_concept
        self.seq_len = 500
        self.num_heads = 8

        self.at_embed = nn.Embedding(n_at + 10, d_k)
        self.it_embed = nn.Embedding(n_it + 10, d_k)
        self.e_embed = nn.Embedding(n_exercise + 10, d_e)
        self.c_embed = nn.Embedding(n_concept + 10, d_k)
        self.linear_all_learning = nn.Linear(d_k + d_a + d_e + d_a, d_k)
        self.linear_now_rec = nn.Linear(4 * d_k, d_k)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.q_embedding_is = nn.Embedding(
            self.n_concept + 1, d_k, padding_idx=self.n_concept
        )
        self.pos_embedding_is = nn.Embedding(self.seq_len, d_k)
        self.multi_attention_is = nn.MultiheadAttention(
            embed_dim=d_k, num_heads=self.num_heads, dropout=dropout
        )
        self.linear_is = nn.ModuleList([nn.Linear(in_features=d_k, out_features=d_k) for x in range(3)])
        self.layer_norm1_is = nn.LayerNorm(d_k)
        self.q_embedding_cs = nn.Embedding(
            self.n_concept + 1, d_k, padding_idx=self.n_concept
        )
        self.pos_embedding_cs = nn.Embedding(self.seq_len, d_k)
        self.multi_attention_cs = nn.MultiheadAttention(
            embed_dim=d_k, num_heads=self.num_heads, dropout=dropout
        )
        self.linear_cs = nn.ModuleList([nn.Linear(in_features=d_k, out_features=d_k) for x in range(3)])
        self.layer_norm1_cs = nn.LayerNorm(d_k)
        self.fc_att_cs_is = nn.Linear(2 * d_k, d_k)
        self.linear_l_it = nn.Linear(2 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_l_it.weight)
        self.linear_l_at = nn.Linear(2 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_l_at.weight)

        self.linear_l_it_h = nn.Linear(2 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_l_it_h.weight)
        self.linear_l_at_h = nn.Linear(2 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_l_at_h.weight)

        self.conv1d = nn.Conv1d(d_k, d_k, 2)

        self.linear_rec2cur = nn.Linear(2 * d_k, d_k)

        self.linear_y = nn.Linear(d_e + d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_y.weight)

    def forward(self, e_data, at_data, a_data, it_data, c_data, ca_data, recent_c, e_diff):
        batch_size, seq_len = e_data.size(0), e_data.size(1)
        e_embed_data = self.e_embed(e_data)
        at_embed_data = self.at_embed(at_data)
        it_embed_data = self.it_embed(it_data)
        c_embed_data = self.c_embed(c_data)
        a_data = a_data.view(-1, 1).repeat(1, self.d_a).view(batch_size, -1, self.d_a)
        e_diff_data = e_diff.view(-1, 1).repeat(1, self.d_a).view(batch_size, -1, self.d_a).float()

        h_pre = nn.init.xavier_uniform_(torch.zeros(self.n_concept + 1, self.d_k)).repeat(batch_size, 1, 1).to(device)
        h_tilde_pre = None

        ## IS-sampling
        all_learning = self.linear_all_learning(torch.cat((e_embed_data, c_embed_data, a_data, e_diff_data), 2))

        save_all_learning = all_learning.view(batch_size * seq_len, -1)

        pos_id = torch.arange(all_learning.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_embedding_is(pos_id)
        all_learning_att_in = all_learning + pos_x
        all_learning_att_in = all_learning_att_in.permute(1, 0, 2)

        query_in = all_learning_att_in
        key_in = all_learning_att_in
        value_in = all_learning_att_in

        value_in = self.linear_is[0](value_in).to(device)
        key_in = self.linear_is[1](key_in).to(device)
        query_in = self.linear_is[2](query_in).to(device)

        attention_mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype("bool")).to(device)
        attention_out, _ = self.multi_attention_is(query_in, key_in, value_in, attn_mask=attention_mask)
        attention_out = self.layer_norm1_is(attention_out + query_in)
        attention_out = attention_out.permute(1, 0, 2)
        
        # CS-sampling
        global_learning = torch.load('./center_learning_2000.pt').to(device)
        global_learning = torch.unsqueeze(global_learning, 0)
        MC = global_learning.repeat(batch_size, 1, 1)
        MC = MC.permute(1, 0, 2)
        # Q' K' V'
        query_in_hat = all_learning_att_in
        key_in_hat = MC
        value_in_hat = MC
        
        value_in_hat = self.linear_cs[0](value_in_hat).to(device)
        key_in_hat = self.linear_cs[1](key_in_hat).to(device)
        query_in_hat = self.linear_cs[2](query_in_hat).to(device)
        
        attention_mask = torch.from_numpy(np.triu(np.ones((seq_len, 2000)), k=1).astype("bool")).to(device)
        attention_out_hat, _ = self.multi_attention_cs(query_in_hat, key_in_hat, value_in_hat, attn_mask=attention_mask)
        attention_out_hat = self.layer_norm1_cs(attention_out_hat + query_in_hat)
        attention_out_hat = attention_out_hat.permute(1, 0, 2)
        
        attention_out = self.fc_att_cs_is(torch.cat([attention_out, attention_out_hat], dim=2))
        #
        all_learning = attention_out

        # learning_pre:(bs, d_k)
        # learning_pre = torch.zeros(batch_size, self.d_k).to(device)

        learning_pre = None
        pred = torch.zeros(batch_size, seq_len).to(device)

 
        for t in range(0, seq_len - 1):
            e = e_data[:, t]
            q_e = self.q_matrix[e].view(batch_size, 1, -1)
            it = it_embed_data[:, t]
            at = at_embed_data[:, t]

            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k)

            learning = all_learning[:, t]
            recent_learning = recent_l[:, t]
            if learning_pre is None:
                learning_pre = torch.zeros(batch_size, self.d_k).to(device)


            learning_at = torch.cat((learning, at), 1)
            learning_at = self.linear_l_at(learning_at)
            learning_at = torch.cat((h_tilde_pre, learning_at), 1)
            learning_at = self.linear_l_at_h(learning_at)
            gamma_l = self.sig(learning_at)
            learning_gain = self.tanh(learning_at)
            l_input_tilde = gamma_l * ((learning_gain + 1) / 2)
            l_input = self.dropout(q_e.transpose(1, 2).bmm(l_input_tilde.view(batch_size, 1, -1)))

            n_concept = l_input.size(1)

            learning_it = torch.cat((learning.repeat(1, n_concept).view(batch_size, -1, self.d_k),
                                     it.repeat(1, n_concept).view(batch_size, -1, self.d_k)), 2)
            learning_it = self.linear_l_it(learning_it)
            learning_it = torch.cat((h_tilde_pre.repeat(1, n_concept).view(batch_size, -1, self.d_k),
                                     learning_it), 2)
            learning_it = self.linear_l_it_h(learning_it)
            gamma_f = self.sig(learning_it)

            h = h_pre * gamma_f + l_input

            # Predicting Module
            h_tilde = self.q_matrix[e_data[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size, self.d_k)
            y = self.sig(self.linear_y(torch.cat((e_embed_data[:, t + 1], h_tilde), 1))).sum(1) / self.d_k
            pred[:, t + 1] = y

            # prepare for next prediction
            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde

        return pred, save_all_learning
