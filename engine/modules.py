import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import List, Tuple, Dict
import numpy as np
from torch.autograd import Variable


class DepGCN(nn.Module):
    """
    Label-aware Dependency Convolutional Neural Network Layer
    """

    def __init__(self, dep_num, dep_dim, in_features, out_features):
        super(DepGCN, self).__init__()
        self.dep_dim = dep_dim
        self.in_features = in_features
        self.out_features = out_features

        self.dep_embedding = nn.Embedding(dep_num, dep_dim, padding_idx=0)

        self.dep_attn = nn.Linear(dep_dim + in_features, out_features)
        self.dep_fc = nn.Linear(dep_dim, out_features)
        self.relu = nn.ReLU()

    def forward(self, text, dep_mat, dep_labels):
        dep_label_embed = self.dep_embedding(dep_labels)

        batch_size, seq_len, feat_dim = text.shape

        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, seq_len, 1)

        val_sum = torch.cat([val_us, dep_label_embed], dim=-1)

        r = self.dep_attn(val_sum)

        p = torch.sum(r, dim=-1)
        mask = (dep_mat == 0).float() * (-1e30)
        p = p + mask
        p = torch.softmax(p, dim=2)
        p_us = p.unsqueeze(3).repeat(1, 1, 1, feat_dim)

        output = val_us + self.dep_fc(dep_label_embed)
        output = torch.mul(p_us, output)

        output_sum = torch.sum(output, dim=2)
        output_sum = self.relu(output_sum)

        return output_sum


class ConstGCN(nn.Module):
    """
    Label-aware Constituency Convolutional Neural Network Layer
    """

    def __init__(
            self,
            num_inputs,
            num_units,
            num_labels,
            dropout=0.0,
            in_arcs=True,
            out_arcs=True,
            batch_first=False,
            use_gates=True,
            residual=False,
            no_loop=False,
            non_linearity="relu",
            edge_dropout=0.0,
    ):
        super(ConstGCN, self).__init__()

        self.in_arcs = in_arcs
        self.out_arcs = out_arcs
        self.no_loop = no_loop
        self.retain = 1.0 - edge_dropout
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.non_linearity = non_linearity
        self.sigmoid = nn.Sigmoid()
        self.use_gates = use_gates
        self.residual = residual
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(num_units)

        if in_arcs:
            self.V_in = Parameter(torch.Tensor(self.num_inputs, self.num_units))
            nn.init.xavier_normal_(self.V_in)

            self.b_in = Parameter(torch.Tensor(num_labels, self.num_units))
            nn.init.constant_(self.b_in, 0)

            if self.use_gates:
                self.V_in_gate = Parameter(torch.Tensor(self.num_inputs, 1))
                nn.init.xavier_normal_(self.V_in_gate)
                self.b_in_gate = Parameter(torch.Tensor(num_labels, 1))
                nn.init.constant_(self.b_in_gate, 1)

        if out_arcs:
            # self.V_out = autograd.Variable(torch.FloatTensor(self.num_inputs, self.num_units))
            self.V_out = Parameter(torch.Tensor(self.num_inputs, self.num_units))
            nn.init.xavier_normal_(self.V_out)

            # self.b_out = autograd.Variable(torch.FloatTensor(num_labels, self.num_units))
            self.b_out = Parameter(torch.Tensor(num_labels, self.num_units))
            nn.init.constant_(self.b_out, 0)

            if self.use_gates:
                self.V_out_gate = Parameter(torch.Tensor(self.num_inputs, 1))
                nn.init.xavier_normal_(self.V_out_gate)
                self.b_out_gate = Parameter(torch.Tensor(num_labels, 1))
                nn.init.constant_(self.b_out_gate, 1)
        if not self.no_loop:
            self.W_self_loop = Parameter(torch.Tensor(self.num_inputs, self.num_units))
            nn.init.xavier_normal_(self.W_self_loop)

            if self.use_gates:
                self.W_self_loop_gate = Parameter(torch.Tensor(self.num_inputs, 1))
                nn.init.xavier_normal_(self.W_self_loop_gate)

    def forward(
            self,
            src,
            arc_tensor_in=None,
            arc_tensor_out=None,
            label_tensor_in=None,
            label_tensor_out=None,
            mask_in=None,
            mask_out=None,
            mask_loop=None,
            sent_mask=None,
    ):

        if not self.batch_first:
            encoder_outputs = src.permute(1, 0, 2).contiguous()
        else:
            encoder_outputs = src.contiguous()

        batch_size = encoder_outputs.size()[0]
        seq_len = encoder_outputs.size()[1]
        max_degree = 1
        input_ = encoder_outputs.view(
            (batch_size * seq_len, self.num_inputs)
        )  # [b* t, h]
        input_ = self.dropout(input_)
        if self.in_arcs:
            input_in = torch.mm(input_, self.V_in)  # [b* t, h] * [h,h] = [b*t, h]
            first_in = input_in.index_select(
                0, arc_tensor_in[0] * seq_len + arc_tensor_in[1]
            )  # [b* t* degr, h]
            second_in = self.b_in.index_select(0, label_tensor_in[0])  # [b* t* degr, h]
            in_ = first_in + second_in
            degr = int(first_in.size()[0] / batch_size // seq_len)
            in_ = in_.view((batch_size, seq_len, degr, self.num_units))
            if self.use_gates:
                # compute gate weights
                input_in_gate = torch.mm(
                    input_, self.V_in_gate
                )  # [b* t, h] * [h,h] = [b*t, h]
                first_in_gate = input_in_gate.index_select(
                    0, arc_tensor_in[0] * seq_len + arc_tensor_in[1]
                )  # [b* t* mxdeg, h]
                second_in_gate = self.b_in_gate.index_select(0, label_tensor_in[0])
                in_gate = (first_in_gate + second_in_gate).view(
                    (batch_size, seq_len, degr)
                )

            max_degree += degr

        if self.out_arcs:
            input_out = torch.mm(input_, self.V_out)  # [b* t, h] * [h,h] = [b* t, h]
            first_out = input_out.index_select(
                0, arc_tensor_out[0] * seq_len + arc_tensor_out[1]
            )  # [b* t* mxdeg, h]
            second_out = self.b_out.index_select(0, label_tensor_out[0])

            degr = int(first_out.size()[0] / batch_size // seq_len)
            max_degree += degr

            out_ = (first_out + second_out).view(
                (batch_size, seq_len, degr, self.num_units)
            )

            if self.use_gates:
                # compute gate weights
                input_out_gate = torch.mm(
                    input_, self.V_out_gate
                )  # [b* t, h] * [h,h] = [b* t, h]
                first_out_gate = input_out_gate.index_select(
                    0, arc_tensor_out[0] * seq_len + arc_tensor_out[1]
                )  # [b* t* mxdeg, h]
                second_out_gate = self.b_out_gate.index_select(0, label_tensor_out[0])
                out_gate = (first_out_gate + second_out_gate).view(
                    (batch_size, seq_len, degr)
                )
        if self.no_loop:
            if self.in_arcs and self.out_arcs:
                potentials = torch.cat((in_, out_), dim=2)  # [b, t,  mxdeg, h]
                if self.use_gates:
                    potentials_gate = torch.cat(
                        (in_gate, out_gate), dim=2
                    )  # [b, t,  mxdeg, h]
                mask_soft = torch.cat((mask_in, mask_out), dim=1)  # [b* t, mxdeg]
            elif self.out_arcs:
                potentials = out_  # [b, t,  2*mxdeg+1, h]
                if self.use_gates:
                    potentials_gate = out_gate  # [b, t,  mxdeg, h]
                mask_soft = mask_out  # [b* t, mxdeg]
            elif self.in_arcs:
                potentials = in_  # [b, t,  2*mxdeg+1, h]
                if self.use_gates:
                    potentials_gate = in_gate  # [b, t,  mxdeg, h]
                mask_soft = mask_in  # [b* t, mxdeg]
            max_degree -= 1
        else:
            same_input = torch.mm(input_, self.W_self_loop).view(
                encoder_outputs.size(0), encoder_outputs.size(1), -1
            )
            same_input = same_input.view(
                encoder_outputs.size(0),
                encoder_outputs.size(1),
                1,
                self.W_self_loop.size(1),
            )
            if self.use_gates:
                same_input_gate = torch.mm(input_, self.W_self_loop_gate).view(
                    encoder_outputs.size(0), encoder_outputs.size(1), -1
                )

            if self.in_arcs and self.out_arcs:
                potentials = torch.cat(
                    (in_, out_, same_input), dim=2
                )  # [b, t,  mxdeg, h]
                if self.use_gates:
                    potentials_gate = torch.cat(
                        (in_gate, out_gate, same_input_gate), dim=2
                    )  # [b, t,  mxdeg, h]
                mask_soft = torch.cat(
                    (mask_in, mask_out, mask_loop), dim=1
                )  # [b* t, mxdeg]
            elif self.out_arcs:
                potentials = torch.cat(
                    (out_, same_input), dim=2
                )  # [b, t,  2*mxdeg+1, h]
                if self.use_gates:
                    potentials_gate = torch.cat(
                        (out_gate, same_input_gate), dim=2
                    )  # [b, t,  mxdeg, h]
                mask_soft = torch.cat((mask_out, mask_loop), dim=1)  # [b* t, mxdeg]
            elif self.in_arcs:
                potentials = torch.cat(
                    (in_, same_input), dim=2
                )  # [b, t,  2*mxdeg+1, h]
                if self.use_gates:
                    potentials_gate = torch.cat(
                        (in_gate, same_input_gate), dim=2
                    )  # [b, t,  mxdeg, h]
                mask_soft = torch.cat((mask_in, mask_loop), dim=1)  # [b* t, mxdeg]
            else:
                potentials = same_input  # [b, t,  2*mxdeg+1, h]
                if self.use_gates:
                    potentials_gate = same_input_gate  # [b, t,  mxdeg, h]
                mask_soft = mask_loop  # [b* t, mxdeg]

        potentials_resh = potentials.view(
            (batch_size * seq_len, max_degree, self.num_units)
        )  # [h, b * t, mxdeg]

        if self.use_gates:
            potentials_r = potentials_gate.view(
                (batch_size * seq_len, max_degree)
            )  # [b * t, mxdeg]
            probs_det_ = (self.sigmoid(potentials_r) * mask_soft).unsqueeze(
                2
            )  # [b * t, mxdeg]

            potentials_masked = potentials_resh * probs_det_  # [b * t, mxdeg,h]
        else:
            # NO Gates
            potentials_masked = potentials_resh * mask_soft.unsqueeze(2)

        if self.retain == 1 or not self.training:
            pass
        else:
            mat_1 = torch.Tensor(mask_soft.data.size()).uniform_(0, 1)
            ret = torch.Tensor([self.retain])
            mat_2 = (mat_1 < ret).float()
            drop_mask = Variable(mat_2, requires_grad=False)
            if potentials_resh.is_cuda:
                drop_mask = drop_mask.cuda()

            potentials_masked *= drop_mask.unsqueeze(2)

        potentials_masked_ = potentials_masked.sum(dim=1)  # [b * t, h]

        potentials_masked_ = self.layernorm(potentials_masked_) * sent_mask.view(
            batch_size * seq_len
        ).unsqueeze(1)

        potentials_masked_ = self.non_linearity(potentials_masked_)  # [b * t, h]

        result_ = potentials_masked_.view(
            (batch_size, seq_len, self.num_units)
        )  # [ b, t, h]

        result_ = result_ * sent_mask.unsqueeze(2)  # [b, t, h]
        memory_bank = result_  # [t, b, h]

        if self.residual:
            memory_bank += src

        return memory_bank


class BilinearScorer(nn.Module):
    def __init__(self, hidden_dim, role_vocab_size, dropout=0.0, gpu_id=-1):
        super(BilinearScorer, self).__init__()

        if gpu_id > -1:
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.hidden_dim = hidden_dim
        self.role_vocab_size = role_vocab_size

        self.dropout = nn.Dropout(p=dropout)

        self.U = Parameter(
            torch.Tensor(self.hidden_dim, self.role_vocab_size, self.hidden_dim)
        )
        nn.init.orthogonal_(self.U)

        self.bias1 = Parameter(torch.Tensor(1, self.hidden_dim * self.role_vocab_size))
        nn.init.constant_(self.bias1, 0)
        self.bias2 = Parameter(torch.Tensor(1, self.role_vocab_size))
        nn.init.constant_(self.bias2, 0)

    def forward(self, pred_input, args_input):

        b, t, h = pred_input.data.shape
        pred_input = self.dropout(pred_input)
        args_input = self.dropout(args_input)

        first = (
            torch.mm(pred_input.view(-1, h), self.U.view(h, -1)) + self.bias1
        )  # [b*t, h] * [h,r*h] = [b*t,r*h]

        out = torch.bmm(
            first.view(-1, self.role_vocab_size, h), args_input.view(-1, h).unsqueeze(2)
        )  # [b*t,r,h] [b*t, h, 1] = [b*t, r]
        out = out.squeeze(2) + self.bias2
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v, attn_mask):
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn_score.masked_fill_(attn_mask, -1e9)

        attn_weights = nn.Softmax(dim=-1)(attn_score)

        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k)
        self.linear = nn.Linear(n_heads * self.d_v, d_model)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)

        q_heads = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_heads = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_heads = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, attn_mask)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(attn)

        return output, attn_weights


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        output = self.relu(self.linear1(inputs))
        output = self.linear2(output)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_drop, d_ff):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, attn_mask):
        attn_outputs, attn_weights = self.mha(inputs, inputs, inputs, attn_mask)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs + attn_outputs)

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)

        return ffn_outputs, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, seq_len=300, d_model=768, n_layers=3, n_heads=8, p_drop=0.1, d_ff=500, pad_id=0):
        super(TransformerEncoder, self).__init__()
        self.pad_id = pad_id
        self.sinusoid_table = self.get_sinusoid_table(seq_len + 1, d_model)  # (seq_len+1, d_model)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)])

    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)

        outputs = self.embedding(inputs) + self.pos_embedding(positions)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)

        for layer in self.layers:
            outputs, attn_weights = layer(outputs, attn_pad_mask)

        return outputs

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)

        return attn_pad_mask

    def get_sinusoid_table(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i // 2)) / d_model)

        sinusoid_table = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

        return torch.FloatTensor(sinusoid_table)


def allowed_transitions(constraint_type: str, labels: Dict[int, str]) -> List[Tuple[int, int]]:
    """
    Given labels and a constraint type, returns the allowed transitions. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.

    Parameters
    ----------
    constraint_type : ``str``, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    labels : ``Dict[int, str]``, required
        A mapping {label_id -> label}. Most commonly this would be the value from
        Vocabulary.get_index_to_token_vocabulary()

    Returns
    -------
    ``List[Tuple[int, int]]``
        The allowed transitions (from_label_id, to_label_id).
    """
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [(start_tag, "START"), (end_tag, "END")]

    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if is_transition_allowed(constraint_type, from_tag, from_entity,
                                     to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed


def is_transition_allowed(constraint_type: str,
                          from_tag: str,
                          from_entity: str,
                          to_tag: str,
                          to_entity: str):
    """
    Given a constraint type and strings ``from_tag`` and ``to_tag`` that
    represent the origin and destination of the transition, return whether
    the transition is allowed under the given constraint type.

    Parameters
    ----------
    constraint_type : ``str``, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    from_tag : ``str``, required
        The tag that the transition originates from. For example, if the
        label is ``I-PER``, the ``from_tag`` is ``I``.
    from_entity: ``str``, required
        The entity corresponding to the ``from_tag``. For example, if the
        label is ``I-PER``, the ``from_entity`` is ``PER``.
    to_tag : ``str``, required
        The tag that the transition leads to. For example, if the
        label is ``I-PER``, the ``to_tag`` is ``I``.
    to_entity: ``str``, required
        The entity corresponding to the ``to_tag``. For example, if the
        label is ``I-PER``, the ``to_entity`` is ``PER``.

    Returns
    -------
    ``bool``
        Whether the transition is allowed under the given ``constraint_type``.
    """
    # pylint: disable=too-many-return-statements
    if to_tag == "START" or from_tag == "END":
        return False

    if constraint_type == "BIOUL":
        if from_tag == "START":
            return to_tag in ('O', 'B', 'U')
        if to_tag == "END":
            return from_tag in ('O', 'L', 'U')
        return any([
            from_tag in ('O', 'L', 'U') and to_tag in ('O', 'B', 'U'),
            from_tag in ('B', 'I') and to_tag in ('I', 'L') and from_entity == to_entity
        ])
    elif constraint_type == "BIO":
        if from_tag == "START":
            return to_tag in ('O', 'B')
        if to_tag == "END":
            return from_tag in ('O', 'B', 'I')
        return any([
            to_tag in ('O', 'B'),
            to_tag == 'I' and from_tag in ('B', 'I') and from_entity == to_entity
        ])
    elif constraint_type == "IOB1":
        if from_tag == "START":
            return to_tag in ('O', 'I')
        if to_tag == "END":
            return from_tag in ('O', 'B', 'I')
        return any([
            to_tag in ('O', 'I'),
            to_tag == 'B' and from_tag in ('B', 'I') and from_entity == to_entity
        ])
    elif constraint_type == "BMES":
        if from_tag == "START":
            return to_tag in ('B', 'S')
        if to_tag == "END":
            return from_tag in ('E', 'S')
        return any([
            to_tag in ('B', 'S') and from_tag in ('E', 'S'),
            to_tag == 'M' and from_tag == 'B' and from_entity == to_entity,
            to_tag == 'E' and from_tag in ('B', 'M') and from_entity == to_entity,
        ])
    else:
        raise IOError("Unknown constraint type: {constraint_type}")


class CRF(torch.nn.Module):
    def __init__(self,
                 num_tags: int,
                 constraints: List[Tuple[int, int]] = None,
                 include_start_end_transitions: bool = True) -> None:
        super().__init__()
        self.num_tags = num_tags

        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))

        if constraints is None:
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(1.)
        else:
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(0.)
            for i, j in constraints:
                constraint_mask[i, j] = 1.

        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        """
        batch_size, sequence_length, num_tags = logits.size()

        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]

        for i in range(1, sequence_length):
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            inner = broadcast_alpha + emit_scores + transition_scores

            alpha = (logsumexp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))

        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        return logsumexp(stops)

    def _joint_likelihood(self,
                          logits: torch.Tensor,
                          tags: torch.Tensor,
                          mask: torch.LongTensor) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, _ = logits.data.shape

        logits = logits.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        for i in range(sequence_length - 1):
            current_tag, next_tag = tags[i], tags[i + 1]

            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]

            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)

            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        last_inputs = logits[-1]  # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def forward(self,
                inputs: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.ByteTensor = None) -> torch.Tensor:
        """
        Computes the log likelihood.
        """
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)

        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(self,
                     logits: torch.Tensor,
                     mask: torch.Tensor) -> List[Tuple[List[int], float]]:

        _, max_seq_length, num_tags = logits.size()

        logits, mask = logits.data, mask.data

        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)

        constrained_transitions = (
            self.transitions * self._constraint_mask[:num_tags, :num_tags] +
            -10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags])
        )
        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = (
                self.start_transitions.detach() * self._constraint_mask[start_tag, :num_tags].data +
                -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            )
            transitions[:num_tags, end_tag] = (
                self.end_transitions.detach() * self._constraint_mask[:num_tags, end_tag].data +
                -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
            )
        else:
            transitions[start_tag, :num_tags] = (-10000.0 *
                                                 (1 - self._constraint_mask[start_tag, :num_tags].detach()))
            transitions[:num_tags, end_tag] = -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())

        best_paths = []
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)

        for prediction, prediction_mask in zip(logits, mask):
            sequence_length = (torch.sum(prediction_mask)).int()

            tag_sequence.fill_(-10000.)
            tag_sequence[0, start_tag] = 0.
            tag_sequence[1:(sequence_length + 1), :num_tags] = prediction[:sequence_length]
            tag_sequence[sequence_length + 1, end_tag] = 0.

            viterbi_path, viterbi_score = viterbi_decode(tag_sequence[:(sequence_length + 2)], transitions)
            viterbi_path = viterbi_path[1:-1]
            best_paths.append((viterbi_path, viterbi_score.item()))

        return best_paths


def logsumexp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def viterbi_decode(tag_sequence: torch.Tensor,
                   transition_matrix: torch.Tensor,
                   tag_observations=None):
    sequence_length, num_tags = list(tag_sequence.size())
    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise IOError("Observations were provided, but they were not the same length "
                          "as the sequence. Found sequence of length: {} and evidence: {}"
                          .format(sequence_length, tag_observations))
    else:
        tag_observations = [-1 for _ in range(sequence_length)]

    path_scores = []
    path_indices = []

    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.
        path_scores.append(one_hot)
    else:
        path_scores.append(tag_sequence[0, :])

    for timestep in range(1, sequence_length):
        summed_potentials = path_scores[timestep - 1].unsqueeze(-1) + transition_matrix
        scores, paths = torch.max(summed_potentials, 0)

        observation = tag_observations[timestep]
        if tag_observations[timestep - 1] != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                print("The pairwise potential between tags you have passed as "
                      "observations is extremely unlikely. Double check your evidence "
                      "or transition potentials!")
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.
            path_scores.append(one_hot)
        else:
            path_scores.append(tag_sequence[timestep, :] + scores.squeeze())
        path_indices.append(paths.squeeze())

    viterbi_score, best_path = torch.max(path_scores[-1], 0)
    viterbi_path = [int(best_path.numpy())]
    for backward_timestep in reversed(path_indices):
        viterbi_path.append(int(backward_timestep[viterbi_path[-1]]))
    viterbi_path.reverse()
    return viterbi_path, viterbi_score
