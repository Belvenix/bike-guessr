import argparse
import logging
import typing as tp

import networkx as nx
import osmnx as ox
import yaml
from config import (
    GRAPHML_TEST_DATA_DIR,
    GRAPHML_TRAIN_DATA_DIR,
    GRAPHML_VALIDATION_DATA_DIR,
)
from encoder import GAE
from torch import Tensor
from tqdm import tqdm


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2137])
    parser.add_argument("--dataset", type=str, default="bikeguessr")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--max_epoch", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--num_features", type=int, default=95,
                        help="number of features in dataset")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--replace_rate", type=float, default=0.0)

    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2,
                        help="`pow`inddex for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")

    parser.add_argument("--max_epoch_f", type=int, default=30)
    parser.add_argument("--lr_f", type=float, default=0.001,
                        help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float,
                        default=0.0, help="weight decay for evaluation")
    parser.add_argument("--linear_prob", action="store_true", default=False)

    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--concat_hidden", action="store_true", default=False)
    parser.add_argument("--path", type=str,
                        default='./data_transformed/bikeguessr.bin')
    parser.add_argument("--eval_epoch", type=int, default=10)
    parser.add_argument("--eval_repeats", type=int, default=5)
    parser.add_argument("--transform", action="store_true")
    parser.add_argument("--targets", nargs='+', default=None)
    parser.add_argument("--wandb_key", type=str, default=None)

    # for graph classification
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--deg4feat", action="store_true",
                        default=False, help="use node degree as input feature")
    parser.add_argument("--batch_size", type=int, default=32)
    
    parser.add_argument("--full_pipline", action="store_true")

    args = parser.parse_args()
    return args


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.safe_load(f)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    logging.info("------ Use best configs ------")
    return args


def build_model(args) -> GAE:
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    out_dim = args.out_dim
    num_layers = args.num_layers
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder
    mask_rate = args.mask_rate
    drop_edge_rate = args.drop_edge_rate
    replace_rate = args.replace_rate

    activation = args.activation
    loss_fn = args.loss_fn
    alpha_l = args.alpha_l
    concat_hidden = args.concat_hidden
    num_features = args.num_features
    model = None

    model = GAE(
        in_dim=num_features,
        num_hidden=num_hidden,
        out_dim=out_dim,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
    )

    return model


def load_graphs(load_train: bool = True, load_test: bool = True, load_validation: bool = True) -> tp.Tuple[tp.List[ox.graph_from_xml], tp.List[ox.graph_from_xml], tp.List[ox.graph_from_xml]]:
    train, test, validation = GRAPHML_TRAIN_DATA_DIR, GRAPHML_TEST_DATA_DIR, GRAPHML_VALIDATION_DATA_DIR,
    train_graph_files, test_graph_files, validation_graph_files = \
        list(train.glob('*.xml')), list(test.glob('*.xml')), list(validation.glob('*.xml'))
    train_graphs = [ox.load_graphml(p) for p in tqdm(train_graph_files, desc='Loading nx train graphs')] \
        if load_train else []
    test_graphs = [ox.load_graphml(p) for p in tqdm(test_graph_files, desc='Loading nx test graphs')] \
        if load_test else []
    validation_graphs = [ox.load_graphml(p) for p in tqdm(validation_graph_files, desc='Loading nx validation graphs')] \
        if load_validation else []
    return train_graphs, test_graphs, validation_graphs


def retrieve_cycle_indices(preds: Tensor) -> tp.Set[int]:
    """Retrieve indices of cycle predictions.

    Args:
        preds (Tensor): Argmaxed predictions tensor whose dimensions are (num_nodes, 1).
    """
    assert len(preds.shape) == 1, 'The tensor must be one-dimensional - (num_nodes, )'
    return set((preds > 0).nonzero().squeeze().tolist())


def fast_retrieve_nx_prediction_graph(graph_networkx: nx.MultiDiGraph, preds: Tensor):
    cycle_indices = retrieve_cycle_indices(preds)
    subgraph = nx.subgraph_view(graph_networkx, filter_edge=lambda u, v, e: int(graph_networkx[u][v][0]['idx']) in cycle_indices)
    return nx.MultiDiGraph(subgraph)
