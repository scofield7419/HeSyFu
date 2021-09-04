import torch
import os
from subprocess import check_output
import copy
import time
import os.path
from engine.utils import get_batch_sup


def evaluate(
    model,
    epoch,
    data,
    val_acc_best,
    word_vec,
    idx_to_roles,
    params,
    modelname,
    save_dir,
    crf,
    adam_stop,
    stop_training,
    bert_tokenizer,
    bert_model,
    eval_type="valid",
    final_eval=False,
    gold_data_file=None,
    gold_file_path=None,
):
    last_time = time.time()

    srl = model
    srl.eval()
    crf.eval()

    if eval_type == "valid":
        print("\nVALIDATION : Epoch {0}".format(epoch))

    batch_size = 16

    all_pred = []
    data_len = len(data)
    for stidx in range(0, data_len, batch_size):

        labels_batch, sentences_batch, predicate_flags_batch, mask_batch, lengths_batch, fixed_embs, \
        dependency_arcs, dependency_labels, constituent_labels, const_GCN_w_c, const_GCN_c_w, const_GCN_c_c, \
        mask_const_batch, predicate_index, bert_embs = get_batch_sup(
            data[stidx : stidx + batch_size],
            word_vec,
            params.gpu_id,
            params.enc_lstm_dim,
            0.0,
            bert_tokenizer,
            bert_model,
        )

        output = srl(
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
                predicate_index,
                bert_embs,
        )
        best_paths = crf.viterbi_tags(output, mask_batch)

        for x, _ in best_paths:
            all_pred += x

    print("Eval took", str(round((time.time() - last_time) / 60, 2)))
    if gold_data_file:
        try:
            annotated_data = _prep_conll_predictions(
                all_pred, gold_data_file, idx_to_roles
            )
            _print_conll_predictions(annotated_data, modelname + "_" + ".txt")

            gold_standard_file = gold_file_path
            precision, recall, eval_acc = _evaluate_conll(
                modelname + "_" + ".txt", gold_standard_file
            )


        except IndexError:
            print(all_pred),
            print(gold_data_file)
            print(idx_to_roles)

    if final_eval:
        print(
            "finalgrep : F1 {0} : {1} precision {0} : {2} recall {0} : {3}".format(
                eval_type, eval_acc, precision, recall
            )
        )
    else:
        print(
            "togrep : results : epoch {0} ; F1 score {1} : {2} "
            "precision {1} : {3} recall {1} : {4}".format(
                epoch, eval_type, eval_acc, precision, recall
            )
        )

    if eval_type == "valid" and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print("saving model at epoch {0}".format(epoch))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, modelname))
            torch.save(crf.state_dict(), os.path.join(save_dir, modelname + "crf"))
            val_acc_best = eval_acc
            adam_stop = 20
            stop_training = False
        else:
            adam_stop -= 1
            if adam_stop == 0:
                stop_training = True
    return eval_acc, val_acc_best, stop_training, adam_stop


def _prep_conll_predictions(pred, conll_gold, idx_to_roles):
    data = copy.deepcopy(conll_gold)
    cur_sent_len = 0
    sent_lenghts = []
    for li, line in enumerate(data):
        if len(line) == 0:
            sent_lenghts.append(cur_sent_len)
            cur_sent_len = 0
        else:
            cur_sent_len += 1
    curr_sent = 0
    line_count = 0
    n_predicates = 0
    prev_open = []
    for li, line in enumerate(data):
        if len(line) == 0:
            if n_predicates > 0:
                line_count += sent_lenghts[curr_sent] * (n_predicates - 1)
                for la, label in enumerate(prev_open):
                    if label != 0:
                        data[li - 1][la + 6] += ")"
            prev_open = []
            curr_sent += 1
        else:
            if len(prev_open) == 0:
                for _ in line[6:]:
                    prev_open.append(0)
            for la, label in enumerate(line[6:]):
                if pred[line_count + (la * sent_lenghts[curr_sent])] >= len(
                    idx_to_roles
                ):
                    lb = "O"
                else:
                    lb = idx_to_roles[pred[line_count + (la * sent_lenghts[curr_sent])]]
                if lb[0] == "O":
                    data[li][la + 6] = "*"
                    if prev_open[la]:
                        data[li - 1][la + 6] += ")"
                    prev_open[la] = 0
                elif lb[0] == "B":
                    if prev_open[la]:
                        data[li - 1][la + 6] += ")"
                    data[li][la + 6] = "(" + lb[2:] + "*"
                    prev_open[la] = lb[2:]
                elif lb[0] == "I":
                    if not prev_open[la]:
                        data[li][la + 6] = "(" + lb[2:] + "*"
                        prev_open[la] = lb[2:]
                    elif lb[2:] != prev_open[la]:
                        data[li - 1][la + 6] += ")"
                        data[li][la + 6] = "(" + lb[2:] + "*"
                        prev_open[la] = lb[2:]
                    else:
                        data[li][la + 6] = "*"
            n_predicates = len(line[6:])

            if n_predicates > 0:
                line_count += 1
    return data


def _print_conll_predictions(data, name):
    with open("data/predictions/" + name, "w") as out:
        for line in data:
            out.write(" ".join(line[5:]) + "\n")


def _evaluate_conll(prediction_file, gold_standard_file):
    script = "data/scripts/srl-eval.pl"

    cut_script_args = ["cut", "-d", " ", "-f", "10-100", gold_standard_file]

    eval_script_args = [
        script,
        "/tmp/" + gold_standard_file.split("/")[-1],
        "data/predictions/" + prediction_file,
    ]

    try:
        DEVNULL = open(os.devnull, "wb")
        cut_out = check_output(cut_script_args, stderr=DEVNULL)
        open("/tmp/" + gold_standard_file.split("/")[-1], "wb").write(cut_out)

        out = check_output(eval_script_args, stderr=DEVNULL)
        out = out.decode("utf-8")

        out_ = " ".join(out.split())
        all_ = out_.split()

        open("data/predictions/" + prediction_file + "_eval.out", "w").write(out)
        prec = all_[27]
        rec = all_[28]
        f1 = all_[29]
        return float(prec), float(rec), float(f1)
    except:
        raise IOError
