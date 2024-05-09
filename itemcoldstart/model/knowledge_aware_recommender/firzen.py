r"""
Firzen
################################################
Reference:
    Hulingxiao He et al. "Firzen: Firing Strict Cold-Start Items with Frozen Heterogeneous and Homogeneous Graphs for Recommendation." in ICDE 2024.
    Paper: http://39.108.48.32/mipl/download_paper.php?fileId=202319 
"""

import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from itemcoldstart.model.abstract_recommender import KnowledgeRecommender
from itemcoldstart.model.init import xavier_normal_initialization
from itemcoldstart.model.loss import BPRLoss, EmbLoss
from itemcoldstart.utils import InputType, ModelType
from collections import defaultdict
from tqdm import tqdm
random.seed(2023)
np.random.seed(2023)

class Aggregator(nn.Module):
    """GNN Aggregator layer"""

    def __init__(self, input_dim, output_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if self.aggregator_type == "gcn":
            self.W = nn.Linear(self.input_dim, self.output_dim)
        elif self.aggregator_type == "graphsage":
            self.W = nn.Linear(self.input_dim * 2, self.output_dim)
        elif self.aggregator_type == "bi":
            self.W1 = nn.Linear(self.input_dim, self.output_dim)
            self.W2 = nn.Linear(self.input_dim, self.output_dim)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()

    def forward(self, norm_matrix, ego_embeddings):
        side_embeddings = torch.sparse.mm(norm_matrix, ego_embeddings)

        if self.aggregator_type == "gcn":
            ego_embeddings = self.activation(self.W(ego_embeddings + side_embeddings))
        elif self.aggregator_type == "graphsage":
            ego_embeddings = self.activation(
                self.W(torch.cat([ego_embeddings, side_embeddings], dim=1))
            )
        elif self.aggregator_type == "bi":
            add_embeddings = ego_embeddings + side_embeddings
            sum_embeddings = self.activation(self.W1(add_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.activation(self.W2(bi_embeddings))
            ego_embeddings = bi_embeddings + sum_embeddings
        else:
            raise NotImplementedError

        ego_embeddings = self.message_dropout(ego_embeddings)

        return ego_embeddings

class Firzen(KnowledgeRecommender):
    r"""Firzen is a unified framework incorporating multi-modal content of items and KGs to effectively solve both strict cold-start and warm-start recommendation termed Firzen, which extracts the user-item collaborative information over frozen heterogeneous graph (collaborative knowledge graph), and exploits the itemitem semantic structures and user-user behavioral association over frozen homogeneous graphs (item-item relation graph and user-user co-occurrence graph).
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, hot_items=None, cold_items=None):
        super(Firzen, self).__init__(config, dataset)

        # define the index of the hot_items and cold_items
        if hot_items is not None:
            self.hot_items = list(hot_items)  # [1, 12100]  0 is [PADDING]
        if cold_items is not None:
            self.cold_items = list(cold_items)  # [3, 12101]

        # KGAT
        # load dataset info
        self.ckg = dataset.ckg_graph(form="dgl", value_field="relation_id")
        self.all_hs = torch.LongTensor(
            dataset.ckg_graph(form="coo", value_field="relation_id").row
        ).to(self.device)
        self.all_ts = torch.LongTensor(
            dataset.ckg_graph(form="coo", value_field="relation_id").col
        ).to(self.device)
        self.all_rs = torch.LongTensor(
            dataset.ckg_graph(form="coo", value_field="relation_id").data
        ).to(self.device)
        self.matrix_size = torch.Size(
            [self.n_users + self.n_entities, self.n_users + self.n_entities]
        )

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.kg_embedding_size = config["kg_embedding_size"]
        self.layers = [self.embedding_size] + config["gnn_layers"]
        self.aggregator_type = config["aggregator_type"]
        self.mess_dropout = config["mess_dropout"]

        # generate intermediate data
        self.A_in = (
            self.init_graph()
        )  # init the attention matrix by the structure of ckg

        self.aggregator_layers = nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(
                zip(self.layers[:-1], self.layers[1:])
        ):
            self.aggregator_layers.append(
                Aggregator(
                    input_dim, output_dim, self.mess_dropout, self.aggregator_type
                )
            )
        self.tanh = nn.Tanh()
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_entity_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_entity_e"]

        # load encoded features here
        self.v_feat, self.t_feat = None, None
        dataset_path = os.path.abspath(config['data_path'])
        # if file exist?
        v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
        t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
        if os.path.isfile(v_feat_file_path):
            self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                self.device)
        if os.path.isfile(t_feat_file_path):
            self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                self.device)

        # load encoded multi-modal features here
        self.m_feat = None
        dataset_path = os.path.abspath(config['data_path'])
        m_feat_file_path = os.path.join(dataset_path, config['mm_feature_file'])
        if os.path.isfile(m_feat_file_path):
            self.m_feat = torch.from_numpy(np.load(m_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                self.device)

        # mask momentum
        self.mask_momentum = 1.0
        self.epochs = config['epochs']

        # load dataset info
        self.dataset = dataset
        self.embedding_size = config["embedding_size"]
        self.feat_embed_size = config["feat_embedding_size"]

        self.kg_embedding_size = config["kg_embedding_size"]
        self.embedding_dim = config['embedding_size']

        self.n_mm_layers = config['n_mm_layers']
        self.sparse = config['sparse']
        self.feat_reg_decay = config['feat_reg_decay']
        self.cl_rate = config['cl_rate']
        self.regs = config['regs']
        self.decay = self.regs[0]

        self.dropout = config['dropout']
        self.model_cat_rate = config['model_cat_rate']
        self.id_cat_rate = config['id_cat_rate']
        self.head_num = config['head_num']

        self.G_rate = config['G_rate']
        self.G_drop1 = config['G_drop1']
        self.G_drop2 = config['G_drop2']
        self.gp_rate = config['gp_rate']
        self.real_data_tau = config['real_data_tau']
        self.ui_pre_scale = config['ui_pre_scale']

        self.T = config['T']
        self.tau = config['tau']
        self.m_topk_rate = config['m_topk_rate']
        self.log_log_scale = config['log_log_scale']

        self.weight_size = config['weight_size']
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        self.batch_size = config['train_batch_size']

        #kg
        self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.trans_w = nn.Embedding(
            self.n_relations, self.embedding_size * self.kg_embedding_size
        )
        self.kg_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # graphs
        # 1. build u-i graph
        self.enable_kg = config["enable_kg"]
        self.ui_graph = self.ui_graph_raw = self.dataset.inter_matrix(form='csr')
        self.iu_graph = self.ui_graph.T
        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]
        self.n_hot_items = len(self.hot_items)
        self.n_cold_items = len(self.cold_items)
        assert self.n_hot_items + self.n_cold_items + 1 == self.n_items, "#hot items + #cold items + 1 = #items should be satisfied."
        # only keep the interaction data of hot items
        self.ui_graph_train = self.matrix_to_tensor(self.csr_norm(self.ui_graph, mean_flag=True))
        self.iu_graph_train = self.matrix_to_tensor(self.csr_norm(self.iu_graph, mean_flag=True))

        # multi-modal encoders
        self.encoder = nn.ModuleDict()
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_size)
            nn.init.xavier_uniform_(self.image_trs.weight)
            self.encoder['image_encoder'] = self.image_trs
            del self.v_feat
            self.v_feat = True
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_size)
            nn.init.xavier_uniform_(self.text_trs.weight)
            self.encoder['text_encoder'] = self.text_trs
            del self.t_feat
            self.t_feat = True
        if self.m_feat is not None:
            self.mm_embedding = nn.Embedding.from_pretrained(self.m_feat, freeze=False)
            self.mm_trs = nn.Linear(self.m_feat.shape[1], self.feat_embed_size)
            nn.init.xavier_uniform_(self.mm_trs.weight)
            self.encoder['mm_encoder'] = self.mm_trs
            del self.m_feat
            self.m_feat = True
        # KG-based item
        if self.enable_kg:
            self.kg_trs = nn.Linear(self.kg_embedding_size, self.feat_embed_size)
            nn.init.xavier_uniform_(self.kg_trs.weight)
            self.encoder['kg_encoder'] = self.kg_trs


        # user&item embedding
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_size)
        # used for knowledge graph-based item embedding
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        nn.init.xavier_uniform_(self.entity_embedding.weight)

        # other paras
        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=self.dropout)
        self.batch_norm = nn.BatchNorm1d(self.embedding_dim)

        # other initialization
        initializer = nn.init.xavier_uniform_
        self.weight_dict = nn.ParameterDict(
            {
                'w_q': nn.Parameter(initializer(torch.empty([self.feat_embed_size, self.feat_embed_size]))),
                'w_k': nn.Parameter(initializer(torch.empty([self.feat_embed_size, self.feat_embed_size]))),
                'w_self_attention_cat': nn.Parameter(
                    initializer(torch.empty([self.head_num * self.feat_embed_size, self.feat_embed_size])))
            }
        )
        self.embedding_dict = {'user': {}, 'item': {}}

        self.n_side_info = 0
        if self.v_feat is not None:
            self.n_side_info += 1
        if self.t_feat is not None:
            self.n_side_info += 1
        if self.m_feat is not None:
            self.n_side_info += 1
        if self.enable_kg:
            self.n_side_info += 1
        self.modality_importance = (
                    torch.ones((self.n_side_info)) / self.n_side_info).cuda()
        self.importance_momentum = config['importance_momentum']


        # discriminator
        self.D = Discriminator(self.n_items, self.G_drop1, self.G_drop2).cuda()
        self.D.apply(self.weights_init)

 
        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']  # not used
        self.n_layers = config['n_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']

        self.batch_size = batch_size
        n_mm_layers = num_user
        self.num_item = num_item
        self.k = 40
        self.adj_momentum = config['adj_momentum']
        self.aggr_mode = config['aggr_mode']
        self.user_aggr_mode = 'softmax'
        #self.num_layer = 1
        self.reg_weight = config['reg_weight']
        self.drop_rate = 0.1
        self.mm_adj = None

        dataset_path = os.path.abspath(config['data_path'])  
        user_graph_path = os.path.join(dataset_path, config['user_graph_dict_file'])

        # construct original i-i graphs
        self.image_adj_file = os.path.join(dataset_path, 'image_adj_{}.pt'.format(self.knn_k))
        self.text_adj_file = os.path.join(dataset_path, 'text_adj_{}.pt'.format(self.knn_k))
        self.mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))
        self.kg_adj_file = os.path.join(dataset_path, 'kg_adj_{}.pt'.format(self.knn_k))

        if self.v_feat is not None:
            if os.path.exists(self.image_adj_file):
                self.image_adj = torch.load(self.image_adj_file)
            else:
                print("Build image i-i adj from scratch ...")
                indices, self.image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                torch.save(self.image_adj, self.image_adj_file)
        if self.t_feat is not None:
            if os.path.exists(self.text_adj_file):
                self.text_adj = torch.load(self.text_adj_file)
            else:
                print("Build text i-i adj from scratch ...")
                indices, self.text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                torch.save(self.text_adj, self.text_adj_file)
        if self.m_feat is not None:
            if os.path.exists(self.mm_adj_file):
                self.mm_adj = torch.load(self.mm_adj_file)
            else:
                print("Build mm i-i adj from scratch ...")
                indices, self.mm_adj = self.get_knn_adj_mat(self.mm_embedding.weight.detach())
                torch.save(self.mm_adj, self.mm_adj_file)
        if self.enable_kg:
            if os.path.exists(self.kg_adj_file):
                self.kg_adj = torch.load(self.kg_adj_file)
            else:
                print("Build kg i-i adj from scratch ...")
                indices, self.kg_adj = self.get_knn_adj_mat(self.entity_embedding.weight[:self.n_items].cuda().detach())
                torch.save(self.kg_adj, self.kg_adj_file)


        # construct the i-i graph for training (masking cold items)
        mask = torch.ones([self.num_item, self.num_item], device=self.device)
        mask[:, list(self.cold_items)] = 0
        mask[list(self.cold_items), :] = 0
        if self.v_feat is not None:
            self.train_image_adj = self.image_adj * mask
        if self.t_feat is not None:
            self.train_text_adj = self.text_adj * mask
        if self.m_feat is not None:
            self.train_mm_adj = self.mm_adj * mask
        if self.enable_kg:
            self.train_kg_adj = self.kg_adj * mask
        del mask

        # build u-u graph
        if os.path.exists(user_graph_path):
            self.user_graph_dict = np.load(user_graph_path, allow_pickle=True).item()
        else:
            user_graph_matrix = self.gen_user_matrix(self.dataset.inter_matrix(form='csr').toarray(), n_mm_layers)
            user_graph = user_graph_matrix
            user_num = torch.zeros(self.num_user)

            user_graph_dict = {}

            for i in range(self.num_user):
                # total number of other users with overlap
                user_num[i] = len(torch.nonzero(user_graph[i]))
                print("this is ", i, "num", user_num[i])

            for i in range(self.num_user):
                if user_num[i] <= 200:
                    # retain all other users that overlap
                    user_i = torch.topk(user_graph[i], int(user_num[i]))
                    edge_list_i = user_i.indices.numpy().tolist()
                    edge_list_j = user_i.values.numpy().tolist()
                    edge_list = [edge_list_i, edge_list_j]
                    user_graph_dict[i] = edge_list

                else:
                    # retain only the 200 other users with the largest number of overlaps
                    user_i = torch.topk(user_graph[i], 200)
                    edge_list_i = user_i.indices.numpy().tolist()
                    edge_list_j = user_i.indices.numpy().tolist()
                    edge_list = [edge_list_i, edge_list_j]
                    user_graph_dict[i] = edge_list
            np.save(os.path.join(dataset_path, config['user_graph_dict_file']), user_graph_dict, allow_pickle=True)

            self.user_graph_dict = np.load(user_graph_path, allow_pickle=True).item()

            del user_graph_matrix

        self.user_graph = User_Graph_sample(num_user, 'add', self.embedding_dim)

        self.restore_user_e = None
        self.restore_item_e = None

        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    def mm(self, x, y):
        if self.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)

    def gen_user_matrix(self, all_edge, no_users):
        edge_dict = defaultdict(set)
        #print(all_edge.shape) 22364 x 12102

        for i in range(all_edge.shape[0]):
            items = np.nonzero(all_edge[i])[0].tolist()
            for item in items:
                edge_dict[i].add(item)

        min_user = 0      # 0
        num_user = no_users   # in our case, users/items ids start from 1
        user_graph_matrix = torch.zeros(num_user, num_user)
        key_list = list(edge_dict.keys())
        key_list.sort()
        bar = tqdm(total=len(key_list))
        for head in range(len(key_list)):
            bar.update(1)
            for rear in range(head + 1, len(key_list)):
                head_key = key_list[head]
                rear_key = key_list[rear]
                item_head = edge_dict[head_key]
                item_rear = edge_dict[rear_key]
                inter_len = len(item_head.intersection(item_rear))
                if inter_len > 0:
                    user_graph_matrix[head_key-min_user][rear_key-min_user] = inter_len
                    user_graph_matrix[rear_key-min_user][head_key-min_user] = inter_len
        bar.close()

        return user_graph_matrix

    def init_graph(self):
        r"""Get the initial attention matrix through the collaborative knowledge graph

        Returns:
            torch.sparse.FloatTensor: Sparse tensor of the attention matrix
        """
        import dgl

        adj_list = []
        for rel_type in range(1, self.n_relations, 1):
            edge_idxs = self.ckg.filter_edges(
                lambda edge: edge.data["relation_id"] == rel_type
            )
            sub_graph = (
                dgl.edge_subgraph(self.ckg, edge_idxs, relabel_nodes=False) #preserve_nodes=True)
                .adjacency_matrix(transpose=False, scipy_fmt="coo")
                .astype("float")
            )
            rowsum = np.array(sub_graph.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(sub_graph).tocoo()
            adj_list.append(norm_adj)

        final_adj_matrix = sum(adj_list).tocoo()
        indices = torch.LongTensor([final_adj_matrix.row, final_adj_matrix.col])
        values = torch.FloatTensor(final_adj_matrix.data)
        adj_matrix_tensor = torch.sparse.FloatTensor(indices, values, self.matrix_size)
        return adj_matrix_tensor.to(self.device)

    def _get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, entity_embeddings], dim=0)
        return ego_embeddings

    def forward_kgat(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.mean(torch.stack(embeddings_list), dim=0)
        user_all_embeddings, entity_all_embeddings = torch.split(
            kgat_all_embeddings, [self.n_users, self.n_entities]
        )
        return user_all_embeddings, entity_all_embeddings

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h).unsqueeze(1)
        pos_t_e = self.entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(
            r.size(0), self.embedding_size, self.kg_embedding_size
        )

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        return h_e, r_e, pos_t_e, neg_t_e

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training rs
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, entity_all_embeddings = self.forward(self.ui_graph_train, self.iu_graph_train)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def calculate_kg_loss(self, interaction):
        r"""Calculate the training loss for a batch data of KG.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """

        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training kg
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_embedding(h, r, pos_t, neg_t)
        pos_tail_score = ((h_e + r_e - pos_t_e) ** 2).sum(dim=1)
        neg_tail_score = ((h_e + r_e - neg_t_e) ** 2).sum(dim=1)
        kg_loss = F.softplus(pos_tail_score - neg_tail_score).mean()
        kg_reg_loss = self.reg_loss(h_e, r_e, pos_t_e, neg_t_e)
        loss = kg_loss + self.reg_weight * kg_reg_loss

        return loss

    def generate_transE_score(self, hs, ts, r):
        r"""Calculating scores for triples in KG.

        Args:
            hs (torch.Tensor): head entities
            ts (torch.Tensor): tail entities
            r (int): the relation id between hs and ts

        Returns:
            torch.Tensor: the scores of (hs, r, ts)
        """

        all_embeddings = self._get_ego_embeddings()
        h_e = all_embeddings[hs].cpu()
        t_e = all_embeddings[ts].cpu()
        r_e = self.relation_embedding.weight[r].cpu()
        r_trans_w = self.trans_w.weight[r].view(
            self.embedding_size, self.kg_embedding_size
        ).cpu()

        h_e = torch.matmul(h_e, r_trans_w)
        t_e = torch.matmul(t_e, r_trans_w)

        kg_score = torch.mul(t_e, self.tanh(h_e + r_e)).sum(dim=1)

        return kg_score

    def update_attentive_A(self):
        r"""Update the attention matrix using the updated embedding matrix"""

        kg_score_list, row_list, col_list = [], [], []
        # To reduce the GPU memory consumption, we calculate the scores of KG triples according to the type of relation
        for rel_idx in range(1, self.n_relations, 1):
            triple_index = torch.where(self.all_rs == rel_idx)
            kg_score = self.generate_transE_score(
                self.all_hs[triple_index], self.all_ts[triple_index], rel_idx
            )
            row_list.append(self.all_hs[triple_index])
            col_list.append(self.all_ts[triple_index])
            kg_score_list.append(kg_score)
        kg_score = torch.cat(kg_score_list, dim=0)
        row = torch.cat(row_list, dim=0)
        col = torch.cat(col_list, dim=0)
        indices = torch.cat([row, col], dim=0).view(2, -1)
        # Current PyTorch version does not support softmax on SparseCUDA, temporarily move to CPU to calculate softmax
        A_in = torch.sparse.FloatTensor(indices.cpu(), kg_score.cpu(), self.matrix_size).cpu()
        self.A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)
        del A_in, kg_score_list, row_list, col_list, kg_score, row, col, indices

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def batched_contrastive_loss(self, z1, z2, batch_size=4096):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size: (i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))
            between_sim = f(self.sim(z1[mask], z2))

            losses.append(-torch.log(
                between_sim[:, i * batch_size: (i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())
            ))

        loss_vec = torch.cat(losses)
        return loss_vec.mean()


    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))
        values = torch.from_numpy(cur_matrix.data)
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()

    def para_dict_to_tensor(self, para_dict):
        """
        Args:
            para_dict: nn.ParameterDict()

        Returns:
            tensor
        """
        tensors = []

        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors

    def multi_head_self_attention(self, trans_w, embedding_t_1, embedding_t):

        q = self.para_dict_to_tensor(embedding_t)
        v = k = self.para_dict_to_tensor(embedding_t_1)
        # d_h: dimension of head
        beh, N, d_h = q.shape[0], q.shape[1], self.feat_embed_size / self.head_num

        Q = torch.matmul(q, trans_w['w_q'])
        K = torch.matmul(k, trans_w['w_k'])
        V = v

        Q = Q.reshape(beh, N, self.head_num, int(d_h)).permute(2, 0, 1, 3)
        K = K.reshape(beh, N, self.head_num, int(d_h)).permute(2, 0, 1, 3)

        Q = torch.unsqueeze(Q, 2)
        K = torch.unsqueeze(K, 1)
        V = torch.unsqueeze(V, 1)

        att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))
        att = torch.sum(att, dim=-1)
        att = torch.unsqueeze(att, dim=-1)
        att = F.softmax(att, dim=2)

        Z = torch.mul(att, V)
        Z = torch.sum(Z, dim=2)

        Z_list = [value for value in Z]
        Z = torch.cat(Z_list, -1)
        Z = torch.matmul(Z, self.weight_dict['w_self_attention_cat'])

        Z = F.normalize(Z, p=2, dim=2)
        return Z, att.detach()

    def gradient_penalty(self, D, xr, xf):

        LAMBDA = 0.3

        xf = xf.detach()
        xr = xr.detach()

        alpha = torch.rand(xr.shape[0], 1).cuda()
        alpha = alpha.expand_as(xr)

        interpolates = alpha * xr + ((1 - alpha) * xf)
        interpolates.requires_grad_()

        disc_interpolates = D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(disc_interpolates),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        return gp

    def forward(self, ui_graph, iu_graph):
        # modality-specific user preference/item representation
        
        if self.v_feat is not None:
            self.image_feats = self.image_embedding.weight
            image_feats = image_item_feats = self.dropout(self.image_trs(self.image_feats))

        if self.t_feat is not None:
            self.text_feats = self.text_embedding.weight
            text_feats = text_item_feats = self.dropout(self.text_trs(self.text_feats))

        if self.m_feat is not None:
            self.mm_feats = self.mm_embedding.weight
            mm_feats = mm_item_feats = self.dropout(self.mm_trs(self.mm_feats))

        if self.enable_kg:
            kg_feats = kg_item_feats = self.dropout(self.kg_trs(self.entity_embedding.weight[:self.n_items]))


        for i in range(self.n_mm_layers):
            if self.v_feat is not None:
                self.image_user_feats = self.mm(ui_graph, image_feats)
                self.image_item_feats = self.mm(iu_graph, self.image_user_feats)

            if self.t_feat is not None:
                self.text_user_feats = self.mm(ui_graph, text_feats)
                self.text_item_feats = self.mm(iu_graph, self.text_user_feats)

            if self.m_feat is not None:
                self.mm_user_feats = self.mm(ui_graph, mm_feats)
                self.mm_item_feats = self.mm(iu_graph, self.mm_user_feats)

            if self.enable_kg:
                self.kg_user_feats = self.mm(ui_graph, kg_feats)
                self.kg_item_feats = self.mm(iu_graph, self.kg_user_feats)

        # CF part
        u_g_embeddings = self.user_embedding.weight
        i_g_embeddings = self.item_id_embedding.weight


        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]
        for i in range(self.n_ui_layers):
            if i == (self.n_ui_layers-1):
                u_g_embeddings = self.softmax(torch.mm(ui_graph, i_g_embeddings))
                i_g_embeddings = self.softmax(torch.mm(iu_graph, u_g_embeddings))

            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings)
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings)

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0)


        # KGAT part

        user_all_embeddings, entity_all_embeddings = self.forward_kgat()
        KG_u_g_embeddings = user_all_embeddings
        KG_i_g_embeddings = entity_all_embeddings[:self.n_items]

        # concatenate KG feats
        
        u_g_embeddings = u_g_embeddings + self.id_cat_rate * F.normalize(KG_u_g_embeddings, p=2, dim=1)
        i_g_embeddings = i_g_embeddings + self.id_cat_rate * F.normalize(KG_i_g_embeddings, p=2, dim=1)


        # concatenate modality feats
        if self.v_feat is not None:
            u_g_embeddings = u_g_embeddings + self.model_cat_rate * self.modality_importance[0] * F.normalize(self.image_user_feats, p=2, dim=1)
            i_g_embeddings = i_g_embeddings + self.model_cat_rate * self.modality_importance[0] * F.normalize(self.image_item_feats, p=2, dim=1)
        if self.t_feat is not None:
            u_g_embeddings = u_g_embeddings + self.model_cat_rate * self.modality_importance[1] * F.normalize(self.text_user_feats, p=2, dim=1)
            i_g_embeddings = i_g_embeddings + self.model_cat_rate * self.modality_importance[1] * F.normalize(self.text_item_feats, p=2, dim=1)
        if self.m_feat is not None:
            u_g_embeddings = u_g_embeddings + self.model_cat_rate * self.modality_importance[0] * F.normalize(self.mm_user_feats, p=2, dim=1)
            i_g_embeddings = i_g_embeddings + self.model_cat_rate * self.modality_importance[0] * F.normalize(self.mm_item_feats, p=2, dim=1)
        if self.enable_kg:
            u_g_embeddings = u_g_embeddings + self.model_cat_rate * self.modality_importance[-1] * F.normalize(self.kg_user_feats, p=2, dim=1)
            i_g_embeddings = i_g_embeddings + self.model_cat_rate * self.modality_importance[-1] * F.normalize(self.kg_item_feats, p=2, dim=1)


        # modality-specific i-i message passing
        # i-i passing

        if self.v_feat is not None:
            image_h = i_g_embeddings.clone()

        if self.t_feat is not None:
            text_h = i_g_embeddings.clone()

        if self.m_feat is not None:
            mm_h = i_g_embeddings.clone()

        if self.enable_kg:
            kg_h = i_g_embeddings.clone()

        for i in range(self.n_layers):
            if self.v_feat is not None:
                image_h = torch.sparse.mm(self.train_image_adj, image_h)
            if self.t_feat is not None:
                text_h =  torch.sparse.mm(self.train_text_adj, text_h)
            if self.m_feat is not None:
                mm_h = torch.sparse.mm(self.train_mm_adj, mm_h)
            if self.enable_kg:
                kg_h = torch.sparse.mm(self.train_kg_adj, kg_h)

        if self.v_feat is not None:
            self.embedding_dict['item']['image'] = image_h
        if self.t_feat is not None:
            self.embedding_dict['item']['text'] = text_h
        if self.m_feat is not None:
            self.embedding_dict['item']['mm'] = mm_h
        if self.enable_kg:
            self.embedding_dict['item']['kg'] = kg_h

        item_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['item'],
                                                   self.embedding_dict['item'])
        item_emb = (self.modality_importance.unsqueeze(-1).unsqueeze(-1) * item_z).sum(0)
        i_g_embeddings += item_emb

        
        # u-u passing
        h_u1 = self.user_graph(u_g_embeddings, self.epoch_user_graph, self.user_weight_matrix)
        u_g_embeddings += h_u1
        
        # restore the warm-start items and user embeddings for test acceleration
        self.result_u_g_embeddings = u_g_embeddings
        self.result_i_g_embeddings = i_g_embeddings

        return u_g_embeddings, i_g_embeddings

    def forward_predict(self, ui_graph, iu_graph, image_adj=None, text_adj=None, mm_adj=None, kg_adj=None):

        if self.v_feat is not None:
            self.image_feats = self.image_embedding.weight
            image_feats = image_item_feats = self.dropout(self.image_trs(self.image_feats))

        if self.t_feat is not None:
            self.text_feats = self.text_embedding.weight
            text_feats = text_item_feats = self.dropout(self.text_trs(self.text_feats))

        if self.m_feat is not None:
            self.mm_item_feats = self.mm_embedding.weight
            mm_feats = mm_item_feats = self.dropout(self.mm_trs(self.mm_item_feats))
        if self.enable_kg:
            kg_feats = kg_item_feats = self.dropout(self.kg_trs(self.entity_embedding.weight[:self.n_items]))


        for i in range(self.n_mm_layers):
            if self.v_feat is not None:
                self.image_user_feats = self.mm(ui_graph, image_feats)
                self.image_item_feats = self.mm(iu_graph, self.image_user_feats)

            if self.t_feat is not None:
                self.text_user_feats = self.mm(ui_graph, text_feats)
                self.text_item_feats = self.mm(iu_graph, self.text_user_feats)

            if self.m_feat is not None:
                self.mm_user_feats = self.mm(ui_graph, mm_feats)
                self.mm_item_feats = self.mm(iu_graph, self.mm_user_feats)

            if self.enable_kg:
                self.kg_user_feats = self.mm(ui_graph, kg_feats)
                self.kg_item_feats = self.mm(iu_graph, self.kg_user_feats)

        # CF part
        u_g_embeddings = self.user_embedding.weight
        i_g_embeddings = self.item_id_embedding.weight

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]
        for i in range(self.n_ui_layers):
            if i == (self.n_ui_layers - 1):
                u_g_embeddings = self.softmax(torch.mm(ui_graph, i_g_embeddings))
                i_g_embeddings = self.softmax(torch.mm(iu_graph, u_g_embeddings))

            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings)
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings)

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0)

        # KGAT part
        user_all_embeddings, entity_all_embeddings = self.forward_kgat()
        KG_u_g_embeddings = user_all_embeddings
        KG_i_g_embeddings = entity_all_embeddings[:self.n_items]

        # concatenate KG feats
        u_g_embeddings = u_g_embeddings + self.id_cat_rate * F.normalize(KG_u_g_embeddings, p=2, dim=1)
        i_g_embeddings = i_g_embeddings + self.id_cat_rate * F.normalize(KG_i_g_embeddings, p=2, dim=1)

        # concatenate modality feats
        if self.v_feat is not None:
            u_g_embeddings = u_g_embeddings + self.model_cat_rate * self.modality_importance[0] * F.normalize(
                self.image_user_feats, p=2, dim=1)
            i_g_embeddings = i_g_embeddings + self.model_cat_rate * self.modality_importance[0] * F.normalize(
                self.image_item_feats, p=2, dim=1)
        if self.t_feat is not None:
            u_g_embeddings = u_g_embeddings + self.model_cat_rate * self.modality_importance[1] * F.normalize(
                self.text_user_feats, p=2, dim=1)
            i_g_embeddings = i_g_embeddings + self.model_cat_rate * self.modality_importance[1] * F.normalize(
                self.text_item_feats, p=2, dim=1)
        if self.m_feat is not None:
            u_g_embeddings = u_g_embeddings + self.model_cat_rate * self.modality_importance[0] * F.normalize(
                self.mm_user_feats, p=2, dim=1)
            i_g_embeddings = i_g_embeddings + self.model_cat_rate * self.modality_importance[0] * F.normalize(
                self.mm_item_feats, p=2, dim=1)
        if self.enable_kg:
            u_g_embeddings = u_g_embeddings + self.model_cat_rate * self.modality_importance[-1] * F.normalize(
                self.kg_user_feats, p=2, dim=1)
            i_g_embeddings = i_g_embeddings + self.model_cat_rate * self.modality_importance[-1] * F.normalize(
                self.kg_item_feats, p=2, dim=1)

        #modality-specific i-i message passing
        #i-i passing
        if self.v_feat is not None:
            image_h = i_g_embeddings.clone()

        if self.t_feat is not None:
            text_h = i_g_embeddings.clone()

        if self.m_feat is not None:
            mm_h = i_g_embeddings.clone()

        if self.enable_kg:
            kg_h = i_g_embeddings.clone()

        for i in range(self.n_layers):
            if self.v_feat is not None:
                image_h = torch.sparse.mm(image_adj, image_h)
            if self.t_feat is not None:
                text_h =  torch.sparse.mm(text_adj, text_h)
            if self.m_feat is not None:
                mm_h = torch.sparse.mm(mm_adj, mm_h)
            if self.enable_kg:
                kg_h = torch.sparse.mm(kg_adj, kg_h)

        if self.v_feat is not None:
            self.embedding_dict['item']['image'] = image_h
        if self.t_feat is not None:
            self.embedding_dict['item']['text'] = text_h
        if self.m_feat is not None:
            self.embedding_dict['item']['mm'] = mm_h
        if self.enable_kg:
            self.embedding_dict['item']['kg'] = kg_h

        item_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['item'],
                                                   self.embedding_dict['item'])
        item_emb = (self.modality_importance.unsqueeze(-1).unsqueeze(-1) * item_z).sum(0)
        i_g_embeddings += item_emb

        # u-u passing
        h_u1 = self.user_graph(u_g_embeddings, self.epoch_user_graph, self.user_weight_matrix)
        u_g_embeddings += h_u1

        return u_g_embeddings, i_g_embeddings

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight).float(), dim=0)  # softmax
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k) / k  # mean
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight).float(), dim=0)  # softmax
            if self.user_aggr_mode == 'mean':
                user_weight_matrix[i] = torch.ones(k) / k  # mean
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def i_sim_calculation(self, user_final, item_final):
        """
        Calculate the sim between cold items and all users
        """
        topk_cold_i = item_final[list(self.cold_items)]

        num_batches = (self.n_users - 1) // self.batch_size + 1
        indices = range(self.n_users)
        u_sim_list = []

        for i_b in range(num_batches):
            index = indices[i_b * self.batch_size: (i_b + 1) * self.batch_size]
            sim = torch.mm(topk_cold_i, user_final[index].T)
            sim_gt = sim
            u_sim_list.append(sim_gt)

        u_sim = F.normalize(torch.cat(u_sim_list, dim=-1), p=2, dim=1)
        return u_sim

    def u_sim_calculation(self, users, user_final, item_final):
        topk_u = user_final[users]

        num_batches = (self.n_items - 1) // self.batch_size + 1
        indices = torch.arange(0, self.n_items).cuda()
        u_sim_list = []

        for i_b in range(num_batches):
            index = indices[i_b * self.batch_size: (i_b + 1) * self.batch_size]
            sim = torch.mm(topk_u, item_final[index].T)
            sim_gt = sim
            u_sim_list.append(sim_gt)

        u_sim = F.normalize(torch.cat(u_sim_list, dim=-1), p=2, dim=1)
        return u_sim

    def calculate_loss_D(self, interaction, idx, total_idx):
        users = interaction[self.USER_ID]

        with torch.no_grad():
            ua_embeddings, ia_embeddings = \
                self.forward(self.ui_graph_train, self.iu_graph_train)

        ui_u_sim_detach = self.u_sim_calculation(users, ua_embeddings, ia_embeddings).detach()
        if self.v_feat is not None:
            image_u_sim_detach = self.u_sim_calculation(users, self.image_user_feats, self.image_item_feats).detach()
            inputf = image_u_sim_detach
        if self.t_feat is not None:
            text_u_sim_detach = self.u_sim_calculation(users, self.text_user_feats, self.text_item_feats).detach()
            inputf = torch.cat((image_u_sim_detach, text_u_sim_detach), dim=0)
        if self.m_feat is not None:
            mm_u_sim_detach = self.u_sim_calculation(users, self.mm_user_feats, self.mm_item_feats).detach()
            inputf = mm_u_sim_detach
        if self.enable_kg:
            kg_u_sim_detach = self.u_sim_calculation(users, self.kg_user_feats, self.kg_item_feats).detach()
            inputf = torch.cat((inputf, kg_u_sim_detach), dim=0)

        predf = (self.D(inputf))
        # try to predict low score for fake
        lossf = (predf.mean())

        new_modality_importance = F.softmax(torch.mean(predf.detach().reshape(self.n_side_info, -1), dim=1), dim=0)
        # update the modality importance
        self.modality_importance = self.importance_momentum * self.modality_importance + new_modality_importance * (
                    1 - self.importance_momentum)

        u_ui = torch.tensor(self.ui_graph_raw[users.cpu()].todense()).cuda()
        # gumbel softmax
        u_ui = F.softmax(u_ui - self.log_log_scale * torch.log(-torch.log(
            torch.empty((u_ui.shape[0], u_ui.shape[1]), dtype=torch.float32).uniform_(0,
                                                                                      1).cuda() + 1e-8) + 1e-8) / self.real_data_tau,
                         dim=-1)
        u_ui += ui_u_sim_detach * self.ui_pre_scale
        u_ui = F.normalize(u_ui, dim=1)

        if self.v_feat is not None:
            inputr = u_ui
        if self.t_feat is not None:
            inputr = torch.cat((u_ui, u_ui), dim=0)
        if self.m_feat is not None:
             inputr = u_ui
        if self.enable_kg:
            inputr = torch.cat((inputr, u_ui), dim=0)
        predr = (self.D(inputr))
        # try to predict high score for real
        lossr = -(predr.mean())

        gp = self.gradient_penalty(self.D, inputr, inputf.detach())

        loss_D = lossr + lossf + self.gp_rate * gp
        return loss_D

    def min_max_normalize(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        range_val = max_val - min_val
        normalized_tensor = (tensor - min_val) / range_val
        return normalized_tensor

    def multiply_with_mask(self, matrix, mask, n):
        masked_matrix = matrix.copy()

        # get sparse matrix slice of selected rows
        selected_sparse_matrix = masked_matrix[n, :]

        masked_matrix[n, :] = selected_sparse_matrix.multiply(mask)

        return masked_matrix


    def calculate_loss_batch(self, interaction, idx, total_idx):
        """
        Args:
            interaction:
            idx: batch idx

        Returns:
        """
        G_batch_mf_loss = 0
        G_batch_emb_loss = 0
        G_batch_reg_loss = 0
        feat_emb_loss = 0
        batch_contrastive_loss = 0 
        G_lossf = 0
        users = interaction[self.USER_ID].cpu()
        pos_items = interaction[self.ITEM_ID].cpu()
        neg_items = interaction[self.NEG_ITEM_ID].cpu()

        G_ua_embeddings, G_ia_embeddings,  \
            = self.forward(self.ui_graph_train, self.iu_graph_train)

        G_u_g_embeddings = G_ua_embeddings[users]
        G_pos_i_g_embeddings = G_ia_embeddings[pos_items]
        G_neg_i_g_embeddings = G_ia_embeddings[neg_items]
        G_batch_mf_loss, G_batch_emb_loss, G_batch_reg_loss = self.bpr_loss(G_u_g_embeddings, G_pos_i_g_embeddings,
                                                                            G_neg_i_g_embeddings)
        if self.v_feat is not None:
            G_image_u_sim = self.u_sim_calculation(users, self.image_user_feats, self.image_item_feats)
            G_image_u_sim_detach = G_image_u_sim.detach()
        if self.t_feat is not None:
            G_text_u_sim = self.u_sim_calculation(users, self.text_user_feats, self.text_item_feats)
            G_text_u_sim_detach = G_text_u_sim.detach()
        if self.m_feat is not None:
            G_mm_u_sim = self.u_sim_calculation(users, self.mm_user_feats, self.mm_item_feats)
            G_mm_u_sim_detach = G_mm_u_sim.detach()
        if self.enable_kg:
            G_kg_u_sim = self.u_sim_calculation(users, self.kg_user_feats, self.kg_item_feats)
            G_kg_u_sim_detach = G_kg_u_sim.detach()


        # in the training phase, we randomly drop virtual user-cold item connections to make the representations more robust

        if self.v_feat is not None:
            side_u_sim_detach = G_image_u_sim_detach * self.modality_importance[0].unsqueeze(-1) #self.mm_weight_dict['w_image']
        if self.t_feat is not None:
            side_u_sim_detach += G_text_u_sim_detach * self.modality_importance[1].unsqueeze(-1) #* self.mm_weight_dict['w_text']
        if self.m_feat is not None:
            side_u_sim_detach = G_mm_u_sim_detach * self.modality_importance[0].unsqueeze(-1) #self.mm_weight_dict['w_mm']
        if self.enable_kg:
            side_u_sim_detach += G_kg_u_sim_detach * self.modality_importance[-1].unsqueeze(-1) #* self.mm_weight_dict['w_kg']

        if self.v_feat is not None:
            feats = [self.image_user_feats, self.image_item_feats]
        if self.t_feat is not None:
            feats.extend([self.text_user_feats, self.text_item_feats])
        if self.m_feat is not None:
            feats = [self.mm_user_feats, self.mm_item_feats]
        if self.enable_kg:
            feats.extend([self.kg_user_feats, self.kg_item_feats])
        feat_emb_loss = self.feat_reg_loss_calculation(*tuple(feats))

        batch_contrastive_loss = 0
        if self.v_feat is not None:
            batch_contrastive_loss += self.batched_contrastive_loss(self.image_user_feats[users],
                                                                    G_ua_embeddings[users], self.batch_size)
            G_inputf = G_image_u_sim_detach
        if self.t_feat is not None:
            batch_contrastive_loss += self.batched_contrastive_loss(self.text_user_feats[users],
                                                                    G_ua_embeddings[users], self.batch_size)
            G_inputf = torch.cat((G_image_u_sim_detach, G_text_u_sim_detach), dim=0)
        if self.m_feat is not None:
            batch_contrastive_loss += self.batched_contrastive_loss(self.mm_user_feats[users],
                                                                    G_ua_embeddings[users], self.batch_size)
            G_inputf = G_mm_u_sim_detach
        # kg
        if self.enable_kg:
            batch_contrastive_loss += self.batched_contrastive_loss(self.kg_user_feats[users],
                                                                    G_ua_embeddings[users], self.batch_size)
            G_inputf = torch.cat((G_inputf, G_kg_u_sim_detach), dim=0)

        G_pref = (self.D(G_inputf))

        G_lossf = -(G_pref.mean())
        batch_loss = G_batch_mf_loss + G_batch_emb_loss + G_batch_reg_loss + feat_emb_loss + self.cl_rate * batch_contrastive_loss + self.G_rate * G_lossf

        return batch_loss

    def feat_reg_loss_calculation(self, *args):
        feat_reg = 0.
        for arg in args:
            feat_reg += 1. / 2 * (arg ** 2).sum()
        feat_reg = feat_reg / self.n_hot_items
        feat_emb_loss = self.feat_reg_decay * feat_reg
        return feat_emb_loss

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum + 1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum + 1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag * csr_mat * colsum_diag
        else:
            return rowsum_diag * csr_mat

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_knn_adj_mat(self, mm_embeddings, add_cold=False):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))

        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def full_sort_predict(self, interaction, cold_items=None, ui_graph_known=None, calc_time=False):
        if not hasattr(self, 'epoch_user_graph'):
            self.pre_epoch_processing()
            
        user = interaction[self.USER_ID]
        mask = torch.ones([self.num_item, self.num_item], device=self.device)
        mask[list(self.hot_items), :] = 0
        mask[:, list(self.cold_items)] = 0
        mask[:, list(self.hot_items)] = 1
        mask[list(self.cold_items), :] = 1


         # only used in normal cold-start item recommendation
        if ui_graph_known is not None:
            ui_graph_test = self.matrix_to_tensor(self.csr_norm(self.ui_graph+ui_graph_known.inter_matrix(form='csr'), mean_flag=True))
            iu_graph_test = self.matrix_to_tensor(self.csr_norm(self.iu_graph+ui_graph_known.inter_matrix(form='csr').T, mean_flag=True))
        else:
            ui_graph_test = self.matrix_to_tensor(self.csr_norm(self.ui_graph, mean_flag=True))
            iu_graph_test = self.matrix_to_tensor(self.csr_norm(self.iu_graph, mean_flag=True))   
            
        if calc_time:  
            curr_time = None         
            if self.restore_user_e is None or self.restore_item_e is None:
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                with torch.no_grad():
                    self.restore_user_e, self.restore_item_e = self.forward_predict(ui_graph_test, iu_graph_test,  self.image_adj * mask if self.v_feat is not None else None, self.text_adj * mask if self.t_feat is not None else None, self.mm_adj * mask if self.m_feat is not None else None, self.kg_adj * mask if self.enable_kg else None)

                ender.record()
                # synchronize GPU time
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) # calculate time
            score_mat_ui = torch.matmul(self.restore_user_e[user], self.restore_item_e.transpose(0, 1))
            return score_mat_ui, curr_time
        else:
            if self.restore_user_e is None or self.restore_item_e is None:
                with torch.no_grad():
                    self.restore_user_e, self.restore_item_e = self.forward_predict(ui_graph_test, iu_graph_test,  self.image_adj * mask if self.v_feat is not None else None, self.text_adj * mask if self.t_feat is not None else None, self.mm_adj * mask if self.m_feat is not None else None, self.kg_adj * mask if self.enable_kg else None)
            score_mat_ui = torch.matmul(self.restore_user_e[user], self.restore_item_e.transpose(0, 1))
        
            return score_mat_ui


class Discriminator(nn.Module):
    def __init__(self, dim, G_drop1, G_drop2):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, int(dim / 4)),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(int(dim / 4)),
            nn.Dropout(G_drop1),

            nn.Linear(int(dim / 4), int(dim / 8)),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(int(dim / 8)),
            nn.Dropout(G_drop2),

            nn.Linear(int(dim / 8), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = 100 * self.net(x.float())
        return output.view(-1)


class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features, user_graph, user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        # pdb.set_trace()
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre