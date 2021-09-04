import torch.autograd as autograd
import torch
from torch.autograd import Variable
import numpy as np


def get_const_adj_BE(batch, max_batch_len, gpu_id, max_degr_in, max_degr_out, forward):
    node1_index = [[word[1] for word in sent] for sent in batch]
    node2_index = [[word[2] for word in sent] for sent in batch]
    label_index = [[word[0] for word in sent] for sent in batch]
    begin_index = [[word[3] for word in sent] for sent in batch]

    batch_size = len(batch)

    _MAX_BATCH_LEN = max_batch_len
    _MAX_DEGREE_IN = max_degr_in
    _MAX_DEGREE_OUT = max_degr_out

    adj_arc_in = np.zeros(
        (batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_IN, 2), dtype="int32"
    )
    adj_lab_in = np.zeros(
        (batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_IN, 1), dtype="int32"
    )
    adj_arc_out = np.zeros(
        (batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_OUT, 2), dtype="int32"
    )
    adj_lab_out = np.zeros(
        (batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_OUT, 1), dtype="int32"
    )

    mask_in = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_IN), dtype="float32")
    mask_out = np.zeros(
        (batch_size * _MAX_BATCH_LEN * _MAX_DEGREE_OUT), dtype="float32"
    )
    mask_loop = np.ones((batch_size * _MAX_BATCH_LEN, 1), dtype="float32")

    tmp_in = {}
    tmp_out = {}

    for d, de in enumerate(node1_index):  # iterates over the batch
        for a, arc in enumerate(de):
            if not forward:
                arc_1 = arc
                arc_2 = node2_index[d][a]
            else:
                arc_2 = arc
                arc_1 = node2_index[d][a]

            if begin_index[d][a] == 0:  # BEGIN
                if arc_1 in tmp_in:
                    tmp_in[arc_1] += 1
                else:
                    tmp_in[arc_1] = 0

                idx_in = (
                    (d * _MAX_BATCH_LEN * _MAX_DEGREE_IN)
                    + arc_1 * _MAX_DEGREE_IN
                    + tmp_in[arc_1]
                )

                if tmp_in[arc_1] < _MAX_DEGREE_IN:
                    adj_arc_in[idx_in] = np.array([d, arc_2])  # incoming arcs
                    adj_lab_in[idx_in] = np.array([label_index[d][a]])  # incoming arcs
                    mask_in[idx_in] = 1.0

            else:  # END
                if arc_1 in tmp_out:
                    tmp_out[arc_1] += 1
                else:
                    tmp_out[arc_1] = 0

                idx_out = (
                    (d * _MAX_BATCH_LEN * _MAX_DEGREE_OUT)
                    + arc_1 * _MAX_DEGREE_OUT
                    + tmp_out[arc_1]
                )

                if tmp_out[arc_1] < _MAX_DEGREE_OUT:
                    adj_arc_out[idx_out] = np.array([d, arc_2])  # outgoing arcs
                    adj_lab_out[idx_out] = np.array(
                        [label_index[d][a]]
                    )  # outgoing arcs
                    mask_out[idx_out] = 1.0

        tmp_in = {}
        tmp_out = {}

    adj_arc_in = torch.LongTensor(np.transpose(adj_arc_in).tolist())
    adj_arc_out = torch.LongTensor(np.transpose(adj_arc_out).tolist())

    adj_lab_in = torch.LongTensor(np.transpose(adj_lab_in).tolist())
    adj_lab_out = torch.LongTensor(np.transpose(adj_lab_out).tolist())

    mask_in = autograd.Variable(
        torch.FloatTensor(
            mask_in.reshape((_MAX_BATCH_LEN * batch_size, _MAX_DEGREE_IN)).tolist()
        ),
        requires_grad=False,
    )
    mask_out = autograd.Variable(
        torch.FloatTensor(
            mask_out.reshape((_MAX_BATCH_LEN * batch_size, _MAX_DEGREE_OUT)).tolist()
        ),
        requires_grad=False,
    )
    mask_loop = autograd.Variable(
        torch.FloatTensor(mask_loop.tolist()), requires_grad=False
    )

    if gpu_id > -1:
        adj_arc_in = adj_arc_in.cuda()
        adj_arc_out = adj_arc_out.cuda()
        adj_lab_in = adj_lab_in.cuda()
        adj_lab_out = adj_lab_out.cuda()
        mask_in = mask_in.cuda()
        mask_out = mask_out.cuda()
        mask_loop = mask_loop.cuda()
    return [
        adj_arc_in,
        adj_arc_out,
        adj_lab_in,
        adj_lab_out,
        mask_in,
        mask_out,
        mask_loop,
    ]



def get_indexes(data, word_to_idx, roles_to_idx):

    for d in data:

        d["i_text"] = [
            word_to_idx[x.lower()] if x.lower() in word_to_idx else word_to_idx["<UNK>"]
            for x in d["text"]
        ]
        d["lower_text"] = [x.lower() for x in d["text"]]

        d["i_labels"] = [
            roles_to_idx[x] if x in roles_to_idx else roles_to_idx["O"]
            for x in d["labels"]
        ]

    return data


def _get_word_dict_GCN(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in sent:
            word = word.lower()
            if word not in word_dict:
                word_dict[word] = ""
    word_dict["<s>"] = ""
    word_dict["</s>"] = ""
    word_dict["<UNK>"] = ""
    return word_dict


def _get_glove_GCN(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    avg = []
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            word, vec = line.split(" ", 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    for word in word_vec:
        if avg == []:
            avg = word_vec[word]
            count = 1
        else:
            avg += word_vec[word]
            count += 1
    word_vec["<UNK>"] = avg / count
    print(
        "Found {0}(/{1}) words with glove vectors".format(len(word_vec), len(word_dict))
    )
    return word_vec


def build_vocab_GCN(sentences, glove_path):
    word_dict = _get_word_dict_GCN(sentences)
    word_vec = _get_glove_GCN(word_dict, glove_path)
    print("Vocab size : {0}".format(len(word_vec)))
    return word_vec


def create_labels_voc(all_data):
    roles_to_idx = {"O": 0}
    idx_to_roles = ["O"]
    for example in all_data:
        for role in example["labels"]:
            if role not in roles_to_idx:
                idx_to_roles.append(role)
                roles_to_idx[role] = len(roles_to_idx)
    return roles_to_idx, idx_to_roles


def create_constraints(all_data, roles_to_idx):
    constraints = set()
    for example in all_data:
        prev_role = ""
        for r, role in enumerate(example["labels"]):
            if r == 0:
                constraints.add((len(roles_to_idx), roles_to_idx[role]))
                if role[0] == "B":
                    constraints.add((0, roles_to_idx[role]))
                    constraints.add((roles_to_idx[role], 0))
                elif role[0] == "I":
                    constraints.add((roles_to_idx[role], 0))

                prev_role = roles_to_idx[role]
            elif r == len(example["labels"]) - 1:
                constraints.add((prev_role, roles_to_idx[role]))
                constraints.add((roles_to_idx[role], len(roles_to_idx) + 1))

                if role[0] == "B":
                    constraints.add((0, roles_to_idx[role]))
                    constraints.add((roles_to_idx[role], 0))
                    constraints.add((len(roles_to_idx), roles_to_idx[role]))
                elif role[0] == "I":
                    constraints.add((roles_to_idx[role], 0))
            else:
                constraints.add((prev_role, roles_to_idx[role]))
                if role[0] == "B":
                    constraints.add((0, roles_to_idx[role]))
                    constraints.add((roles_to_idx[role], 0))
                    constraints.add((len(roles_to_idx), roles_to_idx[role]))
                elif role[0] == "I":
                    constraints.add((roles_to_idx[role], 0))
                    constraints.add((roles_to_idx[role], len(roles_to_idx) + 1))
                prev_role = roles_to_idx[role]

    return list(constraints)


def create_predicate_constraints(all_data, roles_to_idx):
    constraints = {}
    for example in all_data:
        pred = example["predicates"][0].split("#")[1]
        if pred not in constraints:
            constraints[pred] = set()
        for r, role in enumerate(example["labels"]):
            if roles_to_idx[role] not in constraints[pred]:
                constraints[pred].add(roles_to_idx[role])

    return constraints


def create_vocs(train):
    word_to_idx = {"<PAD>": 0, "<UNK>": 1, "<s>": 2, "</s>": 3}

    for example in train:
        for word in example["text"]:
            word = word.lower()
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    return word_to_idx


def read_data(data_file, w_c_to_idx, c_c_to_idx, dep_lb_to_idx):
    data = []
    data_file_ = []
    curr_sent = []
    with open(data_file, "r") as f:
        text = []
        pos = []

        predicate_positions = (
            []
        )  # this contains the positions fo predicate in the sentence
        predicates = []  # this contains disambiguated predicates
        labels = []  # this contains lists of semantic roles, one for each predicate.
        line_count = 0
        inside = []

        dep_head = []
        dep_lb = []

        word_to_constituents = []
        stack_const = []
        stack_num = []
        curr_num = 500

        constituents_to_constituents = []

        children = {}

        for line in f:
            line = " ".join(line.split())
            line_split = line.strip().split()
            curr_sent.append(line_split)

            if len(line_split) > 2:

                text.append(line_split[0])
                pos.append(line_split[1])

                # Adds the word to constituents arcs
                if line_split[2].find("(") > -1:
                    for const in line_split[2].split("(")[1:]:
                        const = const.replace("*", "").replace(")", "")

                        if const == "TOP":
                            pass
                        else:
                            if const not in w_c_to_idx:
                                w_c_to_idx[const] = len(w_c_to_idx)
                            word_to_constituents.append(
                                [w_c_to_idx[const], line_count, curr_num, 0]
                            )

                        stack_num.append(curr_num)
                        stack_const.append(const)
                        curr_num += 1

                if line_split[2].find(")") > -1:
                    for c in line_split[2]:
                        if c == ")":
                            num = stack_num.pop()
                            const = stack_const.pop()

                            if const == "TOP":
                                pass
                            else:
                                if const not in w_c_to_idx:
                                    w_c_to_idx[const] = len(w_c_to_idx)
                                word_to_constituents.append(
                                    [w_c_to_idx[const], line_count, num, 1]
                                )

                            if len(stack_num) != 0:

                                if stack_const[-1] == "TOP":
                                    pass
                                else:

                                    if stack_const[-1] not in c_c_to_idx:
                                        c_c_to_idx[stack_const[-1]] = len(c_c_to_idx)
                                    constituents_to_constituents.append(
                                        [
                                            c_c_to_idx[stack_const[-1]],
                                            stack_num[-1],
                                            num,
                                            0,
                                        ]
                                    )  # from super to sub

                                    if stack_const[-1] not in children:
                                        children[stack_const[-1]] = [const]
                                    else:
                                        children[stack_const[-1]].append(const)

                                if const == "TOP":
                                    pass
                                else:
                                    if const not in c_c_to_idx:
                                        c_c_to_idx[const] = len(c_c_to_idx)
                                    constituents_to_constituents.append(
                                        [c_c_to_idx[const], num, stack_num[-1], 1]
                                    )

                # Adds the dependency
                if line_split[3] != "-":
                    dep_head.append(int(line_split[3]))
                    dep_lb.append(str(line_split[4]))
                    
                    if str(line_split[4]) not in dep_lb_to_idx.keys():
                        dep_lb_to_idx[str(line_split[4])] = len(dep_lb_to_idx)
                        
                    
                # Adds the predicates
                if line_split[5] != "-":
                    predicate_positions.append(line_count)
                    predicates.append(line_split[5] + "." + line_split[4])

                # Adds the roles
                if len(labels) != len(line_split[6:]):
                    for _ in line_split[6:]:
                        labels.append([])
                        inside.append(0)
                for l, label in enumerate(line_split[6:]):
                    if label.find("(") > -1 and label.find(")") > -1:
                        lab = label.split("*")[0][1:]
                        labels[l].append("B-" + lab)

                    elif label.find("(") > -1:
                        if inside[l]:
                            raise OSError("parsing error")
                        else:
                            lab = label.split("*")[0][1:]
                            inside[l] = lab
                            labels[l].append("B-" + lab)
                    elif label.find(")") > -1:
                        if not inside[l]:
                            raise OSError("parsing error")
                        else:
                            labels[l].append("I-" + inside[l])
                            inside[l] = 0
                    else:
                        if inside[l]:
                            labels[l].append("I-" + inside[l])
                        else:
                            labels[l].append("O")

                line_count += 1
            else:
                data_file_ += curr_sent
                curr_sent = []
                if len(predicate_positions) > 0:
                    for p, pred in enumerate(predicates):
                        data.append(
                            {
                                "text": text,
                                "pos": pos,
                                "predicate_position": predicate_positions[p],
                                "predicates": predicates[p],
                                "labels": labels[p],
                                "dep_head": dep_head,
                                "dep_lb": dep_lb,
                                "word_to_constituents": word_to_constituents,
                                "constituents_to_constituents": constituents_to_constituents,
                                "number_constituents": curr_num - 500,
                            }
                        )
                text = []
                pos = []
                word_to_constituents = []
                constituents_to_constituents = []
                children = {}

                dep_head = []
                dep_lb = []

                predicate_positions = []
                predicates = []
                labels = []
                line_count = 0
                inside = []

                stack_const = []
                stack_num = []
                curr_num = 500
    return data, data_file_, w_c_to_idx, c_c_to_idx, dep_lb_to_idx




def read_data_file(data_file):
    data = []
    with open(data_file, "r") as f:
        for line in f:
            line_split = line.strip().split()
            data.append(line_split)
    return data


def get_batch_sup(
    batch, word_vec, gpu_id, lstm_hidden_dim, word_drop, bert_tokenizer, bert_model
):

    max_sent_len = 0
    max_const_len = 0
    lengths = []
    for d in batch:
        max_sent_len = max(len(d["text"]), max_sent_len)
        max_const_len = max(d["number_constituents"], max_const_len)
    batch_len = len(batch)
    sentences = np.zeros((batch_len, max_sent_len))
    predicate_flags = np.zeros((batch_len, max_sent_len))
    labels = np.zeros((batch_len, max_sent_len))
    dependency_arcs = np.zeros((batch_len, max_sent_len, max_sent_len))
    dependency_labels = np.zeros((batch_len, max_sent_len))

    mask = np.zeros((batch_len, max_sent_len))
    fixed_embs = np.zeros((batch_len, max_sent_len, 100))
    bert_embs = np.zeros((batch_len, max_sent_len, 768))
    predicate_index = np.zeros((batch_len, max_sent_len))

    constituent_labels = np.zeros((batch_len, max_const_len, lstm_hidden_dim))

    const_mask = np.zeros((batch_len, max_const_len))

    plain_sentences = []

    for d, data in enumerate(batch):
        num_const = data["number_constituents"]
        const_mask[d][:num_const] = 1.0

        predicate_flags[d, data["predicate_position"]] = 1.0
        for w, word in enumerate(data["i_text"]):
            predicate_index[d, w] = data["predicate_position"]
            if np.random.rand() > word_drop:
                sentences[d, w] = word
            else:
                sentences[d, w] = 1  # UNK
            mask[d, w] = 1.0
            labels[d, w] = data["i_labels"][w]
            word_lower = data["lower_text"][w]

            dependency_labels[d, w] = data["dep_lb"][w]
            dependency_arcs[d, w, w] = 1
            dependency_arcs[d, w, data["dep_head"][w]] = 1
            dependency_arcs[d, data["dep_head"][w], w] = 1

            if word_lower in word_vec:
                fixed_embs[d, w, :] = word_vec[word_lower]
            else:
                fixed_embs[d, w, :] = word_vec["<UNK>"]

        plain_sentences.append(data["text"])
        lengths.append(len(data["i_text"]))

    batch_w_c = []
    for d in batch:
        batch_w_c.append([])
        for i in d["word_to_constituents"]:
            batch_w_c[-1].append([])
            for j in i:
                batch_w_c[-1][-1].append(j)

    batch_c_c = []
    for d in batch:
        batch_c_c.append([])
        for i in d["word_to_constituents"]:
            batch_c_c[-1].append([])
            for j in i:
                batch_c_c[-1][-1].append(j)

    for d, _ in enumerate(batch):
        for t, trip in enumerate(batch_w_c[d]):
            for e, elem in enumerate(trip):
                if elem > 499:
                    batch_w_c[d][t][e] = (elem - 500) + max_sent_len

        for t, trip in enumerate(batch_c_c[d]):
            for e, elem in enumerate(trip):
                if elem > 499:
                    batch_c_c[d][t][e] = (elem - 500) + max_sent_len

    const_GCN_w_c = get_const_adj_BE(
        batch_w_c, max_sent_len + max_const_len, gpu_id, 2, 2, forward=True
    )
    const_GCN_c_w = get_const_adj_BE(
        batch_w_c, max_sent_len + max_const_len, gpu_id, 5, 20, forward=False
    )

    const_GCN_c_c = get_const_adj_BE(
        batch_c_c, max_sent_len + max_const_len, gpu_id, 2, 7, forward=True
    )

    if gpu_id > -1:
        cuda = True
    else:
        cuda = False

    # BERT
    if bert_model is not None:

        bert_encoded_sentences = [
            bert_tokenizer.encode(" ".join(sent), add_special_tokens=True)
            for sent in plain_sentences
        ]
        bert_tokenized_sentences = [
            bert_tokenizer.convert_ids_to_tokens(bert_encoded_sentence)
            for bert_encoded_sentence in bert_encoded_sentences
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [
                torch.tensor(bert_encoded_sentence)
                for bert_encoded_sentence in bert_encoded_sentences
            ],
            padding_value=-1,
            batch_first=True,
        )
        if cuda:
            input_ids = input_ids.cuda()

        attention_mask = input_ids == -1
        input_ids = input_ids.masked_fill(attention_mask, 2)
        attention_mask = (attention_mask.float() - 1).abs()
        if cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        with torch.no_grad():
            bert_last_vectors = bert_model(input_ids, attention_mask=attention_mask)[0]
            if cuda:
                bert_last_vectors = bert_last_vectors.cuda()

        for s, sent in enumerate(bert_last_vectors):
            real_index = 0
            for i, bert_token in enumerate(bert_tokenized_sentences[s]):
                if (
                    bert_token == "</s>"
                    or bert_token == "<s>"
                    or not bert_token.startswith("Ġ")
                ):
                    pass
                else:
                    bert_embs[s, real_index, :] = sent[i, :].cpu()
                    real_index += 1

    labels_batch = _make_VariableLong(labels, cuda, False)
    sentences_batch = _make_VariableLong(sentences, cuda, False)
    predicate_flags_batch = _make_VariableFloat(predicate_flags, cuda, False)
    mask_batch = _make_VariableFloat(mask, cuda, False)
    mask_const_batch = _make_VariableFloat(const_mask, cuda, False)
    lengths_batch = _make_VariableLong(lengths, cuda, False)
    fixed_embs = _make_VariableFloat(fixed_embs, cuda, False)
    bert_embs = _make_VariableFloat(bert_embs, cuda, False)
    constituent_labels = _make_VariableFloat(constituent_labels, cuda, False)
    predicate_index_batch = _make_VariableLong(predicate_index, cuda, False)

    return (
        labels_batch,
        sentences_batch,
        predicate_flags_batch,
        mask_batch,
        lengths_batch,
        fixed_embs,
        dependency_arcs,
        dependency_labels,
        constituent_labels,
        const_GCN_w_c,
        const_GCN_c_w,
        const_GCN_c_c,
        mask_const_batch,
        predicate_index_batch,
        bert_embs,
    )


def _make_VariableFloat(numpy_obj, cuda, requires_grad):
    if cuda:
        return Variable(
            torch.FloatTensor(numpy_obj), requires_grad=requires_grad
        ).cuda()
    else:
        return Variable(torch.FloatTensor(numpy_obj), requires_grad=requires_grad)


def _make_VariableLong(numpy_obj, cuda, requires_grad):
    if cuda:
        return Variable(torch.LongTensor(numpy_obj), requires_grad=requires_grad).cuda()
    else:
        return Variable(torch.LongTensor(numpy_obj), requires_grad=requires_grad)
