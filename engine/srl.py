import torch
import torch.nn as nn
import numpy as np
from engine.modules import ConstGCN, DepGCN, BilinearScorer, TransformerEncoder
from engine.utils import _make_VariableLong


class SRLer(nn.Module):
    def __init__(
            self,
            hidden_dim,
            tagset_size,
            num_layers,
            dep_tag_vocab_size,
            w_c_vocab_size,
            c_c_vocab_size,
            eln,
            use_bert,
            params,
            gpu_id=-1,
    ):
        super(SRLer, self).__init__()
        if gpu_id > -1:
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.num_layers = num_layers
        self.vocab_size = w_c_vocab_size
        self.eln = eln
        self.use_bert = use_bert
        self.params = params
        self.dropout = nn.Dropout(p=params.gcn_dropout)
        self.embedding_dropout = nn.Dropout(p=params.emb_dropout)

        if self.use_bert:
            fixed_dim = 768

        else:
            fixed_dim = 100

        embedding_dim = self.params.emb_dim
        self.indicator_embeddings = nn.Embedding(2, embedding_dim)

        self.tagset_size = tagset_size


        if self.params.non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        elif self.params.non_linearity == "tanh":
            self.non_linearity = nn.Tanh()
        elif self.params.non_linearity == "leakyrelu":
            self.non_linearity = nn.LeakyReLU()
        elif self.params.non_linearity == "celu":
            self.non_linearity = nn.CELU()
        elif self.params.non_linearity == "selu":
            self.non_linearity = nn.SELU()
        else:
            raise NotImplementedError


        self.TrmEncoder = TransformerEncoder(vocab_size=self.vocab_size,
                                             n_layers=self.num_layers,
                                             )
        if self.use_gpu:
            self.TrmEncoder.to(self.device)

        self.hidden2predicate = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2argument = nn.Linear(hidden_dim, hidden_dim)
        self.bilinear_scorer = BilinearScorer(
            hidden_dim, tagset_size, params.bilinear_dropout
        )

        # ConstGCN
        # boundary bridging
        self.const_gcn_w_c = ConstGCN(
            hidden_dim,
            hidden_dim,
            w_c_vocab_size,
            in_arcs=True,
            out_arcs=True,
            use_gates=True,
            batch_first=True,
            residual=True,
            no_loop=True,
            dropout=self.params.gcn_dropout,
            non_linearity=self.non_linearity,
            edge_dropout=self.params.edge_dropout,
        )

        # reverse boundary bridging
        self.const_gcn_c_w = ConstGCN(
            hidden_dim,
            hidden_dim,
            w_c_vocab_size,
            in_arcs=True,
            out_arcs=True,
            use_gates=True,
            batch_first=True,
            residual=True,
            no_loop=True,
            dropout=self.params.gcn_dropout,
            non_linearity=self.non_linearity,
            edge_dropout=self.params.edge_dropout,
        )

        # self graph
        self.const_gcn_c_c = ConstGCN(
            hidden_dim,
            hidden_dim,
            c_c_vocab_size,
            in_arcs=True,
            out_arcs=True,
            use_gates=True,
            batch_first=True,
            residual=True,
            no_loop=False,
            dropout=self.params.gcn_dropout,
            non_linearity=self.non_linearity,
            edge_dropout=self.params.edge_dropout,
        )

        # DepGCN
        self.dep_gcn = DepGCN(dep_tag_vocab_size, hidden_dim + 768, hidden_dim * 2, hidden_dim)

        self.gate = nn.Sigmoid()

        if self.eln:
            self.layernorm = nn.LayerNorm(fixed_dim)

    def forward(
            self,
            sentence,
            predicate_flags,
            sent_mask,
            lengths,
            fixed_embs,
            dependency_arcs,
            dependency_labels,
            constituent_labels,
            const_GCN_w_c,
            const_GCN_c_w,
            const_GCN_c_c,
            mask_const_batch,
            predicate_index,
            bert_embs,
    ):

        if self.use_bert:
            embeds = bert_embs
        else:
            embeds = fixed_embs

        if self.eln:
            embeds = self.layernorm(embeds * sent_mask.unsqueeze(2))

        embeds = self.embedding_dropout(embeds)

        embeds = torch.cat(
            (embeds, self.indicator_embeddings(predicate_flags.long())), 2
        )

        b, t, e = embeds.data.shape
        base_out = self.TrmEncoder(embeds)

        const_gcn_in = torch.cat([base_out, constituent_labels], dim=1)
        mask_all = torch.cat([sent_mask, mask_const_batch], dim=1)

        # boundary bridging
        adj_arc_in_w_c, adj_arc_out_w_c, adj_lab_in_w_c, adj_lab_out_w_c, mask_in_w_c, mask_out_w_c, mask_loop_w_c = (
            const_GCN_w_c
        )

        # inverse-boundary bridging
        adj_arc_in_c_w, adj_arc_out_c_w, adj_lab_in_c_w, adj_lab_out_c_w, mask_in_c_w, mask_out_c_w, mask_loop_c_w = (
            const_GCN_c_w
        )

        adj_arc_in_c_c, adj_arc_out_c_c, adj_lab_in_c_c, adj_lab_out_c_c, mask_in_c_c, mask_out_c_c, mask_loop_c_c = (
            const_GCN_c_c
        )

        const_gcn_out = self.const_gcn_w_c(
            const_gcn_in,
            adj_arc_in_w_c,
            adj_arc_out_w_c,
            adj_lab_in_w_c,
            adj_lab_out_w_c,
            mask_in_w_c,
            mask_out_w_c,
            mask_loop_w_c,
            mask_all,
        )

        const_gcn_out = self.const_gcn_c_c(
            const_gcn_out,
            adj_arc_in_c_c,
            adj_arc_out_c_c,
            adj_lab_in_c_c,
            adj_lab_out_c_c,
            mask_in_c_c,
            mask_out_c_c,
            mask_loop_c_c,
            mask_all,
        )

        const_gcn_out = self.const_gcn_c_w(
            const_gcn_out,
            adj_arc_in_c_w,
            adj_arc_out_c_w,
            adj_lab_in_c_w,
            adj_lab_out_c_w,
            mask_in_c_w,
            mask_out_c_w,
            mask_loop_c_w,
            mask_all,
        )

        # const_gcn_out = const_gcn_out.narrow(1, 0, t)
        dep_gcn_in = torch.cat([base_out, const_gcn_out], dim=1)

        # learn from dependency
        dep_gcn_out = self.dep_gcn(dep_gcn_in, dependency_arcs, dependency_labels)

        # gating
        if self.use_gpu:
            gpu_id = 1
        else:
            gpu_id = 0
        gate_ = self.gate(torch.cat([dep_gcn_out, const_gcn_out], dim=1))
        all_one = _make_VariableLong(np.zeros((b, t)), gpu_id, False)
        hesyfu_out = gate_ * dep_gcn_out + (all_one - gate_) * const_gcn_out

        hesyfu_out_view = hesyfu_out.contiguous().view(b * t, -1)
        predicate_index = predicate_index.view(b * t)

        predicates_repr = hesyfu_out_view.index_select(0, predicate_index).view(b, t, -1)

        pred_repr = self.non_linearity(
            self.hidden2predicate(self.dropout(predicates_repr))
        )
        arg_repr = self.non_linearity(self.hidden2argument(self.dropout(hesyfu_out_view)))
        tag_scores = self.bilinear_scorer(pred_repr, arg_repr)  # [b*t, label_size]

        return tag_scores.view(b, t, self.tagset_size)
