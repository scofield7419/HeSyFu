import os
import torch
import numpy as np
import argparse
from engine.utils import read_data, create_vocs, create_labels_voc, build_vocab_GCN, get_indexes
from evaluation import evaluate
from itertools import chain
from engine.srl import SRLer
from engine.modules import CRF
from pytorch_transformers import RobertaTokenizer, RobertaModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HeSyFu inference")
    parser.add_argument("--dir", type=str, default="savedir/", help="Output directory")
    parser.add_argument("--modelname", type=str, default="model.pickle")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--emb-dim", type=int, default=300, help="word embedding dimension"
    )
    parser.add_argument(
        "--use-roberta", type=int, default=0, help="default do not use RoBERTa embeddings"
    )

    parser.add_argument(
        "--embedding-layer-norm",
        type=int,
        default=0,
        help="default do not embedding layer norm",
    )

    parser.add_argument(
        "--rep_dim", type=int, default=350, help="const_gcn/dep_gcn dimension"
    )
    parser.add_argument("--n-layers", type=int, default=2, help="encoder num layers")

    parser.add_argument(
        "--train-file", type=str, required=True, help="path of the training file"
    )
    parser.add_argument(
        "--dev-file", type=str, required=True, help="path of the training file"
    )

    parser.add_argument(
        "--test-file", type=str, required=True, help="path of the training file"
    )

    parser.add_argument(
        "--glove-path",
        type=str,
        required=True,
        help="path of Glove glove.6B.300d.txt embeddings",
    )

    parser.add_argument(
        "--bilinear-dropout",
        type=float,
        default=0.0,
        help="dropout at the bilinear module",
    )
    parser.add_argument(
        "--gcn-dropout", type=float, default=0.1, help="dropout of the const_gcn/dep_gcn input module"
    )
    parser.add_argument(
        "--emb-dropout",
        type=float,
        default=0.2,
        help="dropout of the embedding , default off",
    )
    parser.add_argument(
        "--edge-dropout",
        type=float,
        default=0.1,
        help="dropout of the const_gcn/dep_gcn edges , default off",
    )
    parser.add_argument(
        "--label-dropout",
        type=float,
        default=0.1,
        help="dropout of the const_gcn/dep_gcn labels , default off",
    )
    parser.add_argument(
        "--non-linearity",
        type=str,
        default="relu",
        help="nonlinearity used, default relu",
    )

    # gpu
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID")
    parser.add_argument("--seed", type=int, default=3070, help="seed")

    params, _ = parser.parse_known_args()

    # set gpu device
    torch.cuda.set_device(params.gpu_id)

    """
    SEED
    """
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    GLOVE_PATH = params.glove_path

    train_file = params.train_file
    dev_file = params.dev_file
    test_file = params.test_file

    train, train_data_file, w_c_to_idx, c_c_to_idx, dep_lb_to_idx = read_data(train_file, {}, {}, {})
    dev, dev_data_file, w_c_to_idx, c_c_to_idx, dep_lb_to_idx = read_data(
        dev_file, w_c_to_idx, c_c_to_idx, dep_lb_to_idx
    )
    test, test_data_file, _, _ = read_data(test_file, w_c_to_idx, c_c_to_idx, dep_lb_to_idx)

    word_to_idx = create_vocs(train)

    roles_to_idx, idx_to_roles = create_labels_voc(train + dev)
    word_vec = build_vocab_GCN(
        [t["text"] for t in train]
        + [t["text"] for t in dev]
        + [t["text"] for t in test],
        GLOVE_PATH,
    )

    test = get_indexes(test, word_to_idx, roles_to_idx)

    srl = SRLer(
        params.rep_dim,
        len(roles_to_idx),
        params.n_layers,
        len(dep_lb_to_idx),
        len(w_c_to_idx),
        len(c_c_to_idx),
        params.embedding_layer_norm,
        params.use_bert,
        params,
        params.gpu_id,
    )

    print(srl)

    crf = CRF(
        len(roles_to_idx), None, include_start_end_transitions=True
    )
    print(crf)

    model_parameters = filter(
        lambda p: p.requires_grad, chain(srl.parameters(), crf.parameters())
    )

    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Total parameters =", num_params)
    print(params)

    if params.use_bert:
        bert_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        bert_model = RobertaModel.from_pretrained(
            "roberta-base", output_hidden_states=True
        )
        if params.gpu_id > -1:
            bert_model.cuda()
    else:
        bert_tokenizer = None
        bert_model = None
    if params.gpu_id > -1:
        srl.cuda()
        crf.cuda()

    srl.load_state_dict(torch.load(os.path.join(params.dir, params.modelname)))

    crf.load_state_dict(torch.load(os.path.join(params.dir, params.modelname + "crf")))

    evaluate(
        srl,
        1000,
        test,
        -1,
        word_vec,
        idx_to_roles,
        params,
        params.modelname,
        params.dir,
        crf,
        20,
        False,
        bert_tokenizer,
        bert_model,
        eval_type="test",
        final_eval=True,
        gold_data_file=test_data_file,
        gold_file_path=test_file,
    )
