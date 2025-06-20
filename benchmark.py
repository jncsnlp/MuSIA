#!/usr/bin/env python
import argparse
import torch
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.covariance import EmpiricalCovariance
from os.path import basename, splitext
from scipy.special import logsumexp
import pandas as pd
import pickle

from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("fc", help="Path to config")
    parser.add_argument("id_train_feature", help="Path to data")
    parser.add_argument("id_val_feature", help="Path to output file")
    parser.add_argument("ood_features", nargs="+", help="Path to ood features")
    parser.add_argument("model", help="Path to model")
    parser.add_argument(
        "--train_label",
        default="datalists/imagenet2012_train_random_200k.txt",
        help="Path to train labels",
    )
    parser.add_argument("--clip_quantile", default=0.99, help="Clip quantile to react")
    parser.add_argument(
        "--clip_quantile_local", default=0.9995, help="Clip quantile to react*"
    )
    parser.add_argument(
        "--gamma", default=0.1, help="hyperparameter in generalized entropy"
    )
    parser.add_argument(
        "--M", default=100, help="Top M classes is used to calculate score"
    )
    parser.add_argument("--batch", type=int, default=1, help="Path to data")  # 256
    parser.add_argument("--workers", type=int, default=4, help="Path to data")
    parser.add_argument("--cfg", default="vit-base-p16-384.py", help="Path to config")

    return parser.parse_args()


# region Helper
def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.0
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh


def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh


def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf
    )

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out


def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def generalized_entropy(softmax_id_val, gamma, M):
    probs = softmax_id_val
    probs_sorted = np.sort(probs, axis=1)[:, -M:]
    scores = np.sum(probs_sorted**gamma * (1 - probs_sorted) ** (gamma), axis=1)

    return -scores


def shannon_entropy(softmax_id_val):
    probs = softmax_id_val
    scores = np.sum(probs * np.log(probs), axis=1)
    return scores


def gradnorm(x, w, b):
    fc = torch.nn.Linear(*w.shape[::-1])
    fc.weight.data[...] = torch.from_numpy(w)
    fc.bias.data[...] = torch.from_numpy(b)
    fc.cuda()

    x = torch.from_numpy(x).float().cuda()
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()

    confs = []

    for i in tqdm(x):
        targets = torch.ones((1, 1000)).cuda()
        fc.zero_grad()
        loss = torch.mean(torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
        # i = i[None] 将形状为 (n,) 的tensor，变为 (1, n)形状的tensor
        loss.backward()
        layer_grad_norm = torch.sum(torch.abs(fc.weight.grad.data)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)


def SHE(feature, softmax, mean_feature_train):
    scores = []
    for i in tqdm(range(feature.shape[0])):
        index = np.argmax(softmax[i])
        scores.append(np.sum(np.multiply(feature[i], mean_feature_train[index])))

    return np.array(scores)


def pNML(x_t_x_inv, features, softmaxs_fn):
    x_proj = np.abs(
        np.matmul(
            np.matmul(np.expand_dims(features, axis=1), x_t_x_inv),
            np.expand_dims(features, axis=-1),
        ).squeeze(-1)
    )
    x_t_g = x_proj / (1 + x_proj)

    n_class = softmaxs_fn.shape[-1]

    nf = np.sum(
        softmaxs_fn / (softmaxs_fn + (1 - softmaxs_fn) * (softmaxs_fn**x_t_g)),
        axis=-1,
    )
    regret = np.log(nf) / np.log(n_class)

    return regret


def NECO(pca_estimator, ss, feature_data, logit_data, model, neco_dim):
    if model in ["deit", "swin"]:
        complete_vectors = feature_data

    complete_vectors = ss.transform(feature_data)

    cls_reduced_all = pca_estimator.transform(complete_vectors)

    score_maxlogit = logit_data.max(axis=-1)

    cls_reduced = cls_reduced_all[:, :neco_dim]

    score = []

    for i in range(cls_reduced.shape[0]):
        sc_complet = LA.norm((complete_vectors[i, :]))
        sc = LA.norm(cls_reduced[i, :])
        sc_finale = sc / sc_complet
        score.append(sc_finale)

    score = np.array(score)

    if model != "resnet":
        score *= score_maxlogit

    return score


def main():
    args = parse_args()
    recall = 0.95

    ood_names = [splitext(basename(ood))[0] for ood in args.ood_features]
    print(f"\nood datasets: {ood_names}")

    print("\nload train labels")
    train_labels = np.array(
        [
            int(line.rsplit(" ", 1)[-1])
            for line in open(args.train_label, "r").readlines()
        ],
        dtype=int,
    )
    print(f"\n{train_labels.shape= }")

    print("\nload w and b")
    w, b = pickle.load(file=open(args.fc, "rb"))
    print(f"\t{w.shape=}, {b.shape=}")

    print("\nload features")
    feature_id_train = pickle.load(open(args.id_train_feature, "rb")).squeeze()
    feature_id_val = pickle.load(open(args.id_val_feature, "rb")).squeeze()
    feature_oods = {
        name: pickle.load(open(features, "rb")).squeeze()
        for name, features in zip(ood_names, args.ood_features)
    }
    print(f"\t{feature_id_train.shape=} ")
    print(f"\t{feature_id_val.shape=} ")
    for name, feature_ood in feature_oods.items():
        print(f"\t\t{name}: {feature_ood.shape}")

    print("\ncomputing logits...")
    logit_id_train = feature_id_train @ w.T + b
    logit_id_val = feature_id_val @ w.T + b
    logit_oods = {name: feat @ w.T + b for name, feat in feature_oods.items()}
    print(f"\t{logit_id_train.shape=} ")
    print(f"\t{logit_id_val.shape=}")
    for name, logit_ood in logit_oods.items():
        print(f"\t\t{name}: {logit_ood.shape}")

    print("\ncomputing softmax...")
    softmax_id_train = softmax(logit_id_train, axis=-1)
    softmax_id_val = softmax(logit_id_val, axis=-1)
    softmax_oods = {name: softmax(logit, axis=-1) for name, logit in logit_oods.items()}
    print(f"\t{softmax_id_train.shape=} ")
    print(f"\t{softmax_id_val.shape=} ")
    for name, softmax_ood in softmax_oods.items():
        print(f"\t\t{name}= {softmax_ood.shape}")

    u = -np.matmul(np.linalg.pinv(w), b)

    df = pd.DataFrame(columns=["method", "oodset", "auroc", "fpr"])
    dfs = []
    fw = open(f"result/all/{args.model}.txt", "w")

    # ---------------------------------------
    method = "NECO"
    print(f"\n{method},{args.model}")

    if args.model == "resnet50d":
        DIM = 400
    if args.model == "repvgg":
        DIM = 100
    if args.model == "swin":
        DIM = 400
    if args.model == "deit":
        DIM = 200
    if args.model == "bit":
        DIM = 100
    if args.model == "vit":
        DIM = 200
    print(f"\t{DIM= }")

    print("\tcomputing pca_estimator")
    ss = StandardScaler()
    complete_vectors_train = ss.fit_transform(feature_id_train)
    pca_estimator = PCA(feature_id_train.shape[1])
    _ = pca_estimator.fit_transform(complete_vectors_train)

    score_id = NECO(pca_estimator, ss, feature_id_val, logit_id_val, args.model, DIM)

    for name, feature_ood in feature_oods.items():
        score_ood = NECO(pca_estimator, ss, feature_ood, logit_oods[name], args.model, DIM)

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

    # ---------------------------------------
    method = "element-mean"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    score_id = logsumexp(logit_id_val, axis=-1) * (1 + np.mean(feature_id_val, axis=-1))

    for name, feature_ood in feature_oods.items():
        score_ood = logsumexp(logit_oods[name], axis=-1) * (
            1 + np.mean(feature_ood, axis=-1)
        )

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")

    # ---------------------------------------
    method = "MuSIA"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    if args.model == "resnet50d":
        m, gamma, DIM = 7, 0.3, 1800
    if args.model == "repvgg":
        m, gamma, DIM = 3, 0.2, 2400
    if args.model == "swin":
        m, gamma, DIM = 9, 0.7, 900
    if args.model == "deit":
        m, gamma, DIM = 1, 0.7, 704
    if args.model == "bit":
        m, gamma, DIM = 9, 0.2, 1400
    if args.model == "vit":
        m, gamma, DIM = 19, 0.2, 640
    print(f"\t{DIM= } , {m=} , {gamma=} , {args.model}")
    fw.write(f"\t{args.model} DIM={DIM}, m={m}, gamma={gamma}\n")

    print("\tcomputing principal space...")
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    # cat_id = np.linalg.norm(np.matmul(feature_id_val - u, NS), axis=-1)
    # cdt_id = np.sum((((1 - softmax_id_val) / softmax_id_val) ** gamma), axis=1)
    # alpha_id = np.linalg.norm(np.sort(feature_id_val, axis=1)[:, -m:], axis=1, ord=1)
    score_id = np.sum((((1 - softmax_id_val) / softmax_id_val) ** gamma), axis=1) / (
        np.linalg.norm(np.matmul(feature_id_val - u, NS), axis=-1)
        * np.linalg.norm(np.sort(feature_id_val, axis=1)[:, -m:], axis=1, ord=1)
    )

    for name, feature_ood in feature_oods.items():
        # cat_ood = np.linalg.norm(np.matmul(feature_ood - u, NS), axis=-1)
        # cdt_ood = np.sum((((1 - softmax_oods[name]) / softmax_oods[name]) ** gamma), axis=1)
        # alpha_ood = np.linalg.norm(np.sort(feature_ood, axis=1)[:, -m:], axis=1, ord=1)
        score_ood = np.sum(
            (((1 - softmax_oods[name]) / softmax_oods[name]) ** gamma), axis=1
        ) / (
            np.linalg.norm(np.matmul(feature_ood - u, NS), axis=-1)
            * np.linalg.norm(np.sort(feature_ood, axis=1)[:, -m:], axis=1, ord=1)
        )

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")

    # ---------------------------------------
    method = "pNML"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    print("Compute the pseudo-inverse of the training feature covariance matrix")
    x_t_x_inv = np.linalg.pinv(
        np.dot(feature_id_train.T, feature_id_train), rcond=1e-15
    )

    print("computing scores...")
    score_id = -pNML(x_t_x_inv, feature_id_val, softmax_id_val)
    for name, feature_ood in feature_oods.items():
        score_ood = -pNML(x_t_x_inv, feature_ood, softmax_oods[name])

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")

    # ---------------------------------------
    method = "SHE"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    print("\tcomputing sorted patterns...")
    pred_labels_train = np.argmax(softmax_id_train, axis=-1)
    mean_feature_train = [
        feature_id_train[pred_labels_train == i].mean(axis=0) for i in tqdm(range(1000))
    ]

    score_id = SHE(feature_id_val, softmax_id_val, mean_feature_train)
    for name, feature_ood in feature_oods.items():
        score_ood = SHE(feature_ood, softmax_oods[name], mean_feature_train)

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")

    # ---------------------------------------
    method = "MSP"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    score_id = softmax_id_val.max(axis=-1)
    for name, softmax_ood in softmax_oods.items():
        score_ood = softmax_ood.max(axis=-1)

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")

    # ---------------------------------------
    method = "Energy"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    score_id = logsumexp(logit_id_val, axis=-1)
    for name, logit_ood in logit_oods.items():
        score_ood = logsumexp(logit_ood, axis=-1)

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")

    # ---------------------------------------
    method = "MaxLogit"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    score_id = logit_id_val.max(axis=-1)
    for name, logit_ood in logit_oods.items():
        score_ood = logit_ood.max(axis=-1)

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")

    # ---------------------------------------
    method = "ReAct + Energy ===> ReAct(ViM)"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    clip = np.quantile(feature_id_train, args.clip_quantile)
    print(f"\tclip quantile {args.clip_quantile}, clip {clip:.4f}")

    logit_id_val_clip = np.clip(feature_id_val, a_min=None, a_max=clip) @ w.T + b
    score_id = logsumexp(logit_id_val_clip, axis=-1)
    for name, feature_ood in feature_oods.items():
        logit_ood_clip = np.clip(feature_ood, a_min=None, a_max=clip) @ w.T + b
        score_ood = logsumexp(logit_ood_clip, axis=-1)

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")

    # ---------------------------------------
    method = "Residual"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    DIM = 1000 if feature_id_val.shape[-1] >= 2048 else 512
    print(f"{DIM=}")

    print("\tcomputing principal space...")
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    score_id = -np.linalg.norm(np.matmul(feature_id_val - u, NS), axis=-1)
    for name, logit_ood, feature_ood in zip(
        ood_names, logit_oods.values(), feature_oods.values()
    ):
        score_ood = -np.linalg.norm(np.matmul(feature_ood - u, NS), axis=-1)

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")

    # # ---------------------------------------
    method = "ViM"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    DIM = 1000 if feature_id_val.shape[-1] >= 2048 else 512
    print(f"{DIM=}")

    print("\tcomputing principal space...")
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    print("\tcomputing alpha...")
    vlogit_id_train = np.linalg.norm(np.matmul(feature_id_train - u, NS), axis=-1)
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
    print(f"\t{alpha=:.4f}")

    vlogit_id_val = np.linalg.norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    score_id = -vlogit_id_val + energy_id_val

    for name, logit_ood, feature_ood in zip(
        ood_names, logit_oods.values(), feature_oods.values()
    ):
        energy_ood = logsumexp(logit_ood, axis=-1)
        vlogit_ood = np.linalg.norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
        score_ood = -vlogit_ood + energy_ood
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")

    # ---------------------------------------
    method = "Shannon entropy"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    score_id = shannon_entropy(softmax_id_val)
    for name, softmax_ood in softmax_oods.items():
        score_ood = shannon_entropy(softmax_ood)

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")

    # ---------------------------------------
    method = "Generalized entropy (GEN)"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    score_id = generalized_entropy(softmax_id_val, args.gamma, args.M)
    for name, softmax_ood in softmax_oods.items():
        score_ood = generalized_entropy(softmax_ood, args.gamma, args.M)

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")

    # -------------------------------------------
    method = "Energy + Local React ==> Local React"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    clip = np.quantile(feature_id_val, q=args.clip_quantile_local)
    print(f"\tclip quantile local {args.clip_quantile_local}, clip {clip:.4f}")

    logit_id_val_clip = np.clip(feature_id_val, a_min=None, a_max=clip) @ w.T + b
    score_id = logsumexp(logit_id_val_clip, axis=-1)
    for name, feature_ood in feature_oods.items():
        clip_ood = np.quantile(feature_ood, q=args.clip_quantile_local)
        logit_ood_clip = np.clip(feature_ood, a_min=None, a_max=clip_ood) @ w.T + b
        score_ood = logsumexp(logit_ood_clip, axis=-1)

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")

    # ---------------------------------------
    method = "GradNorm"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    score_id = gradnorm(feature_id_val, w, b)
    for name, feature_ood in feature_oods.items():
        score_ood = gradnorm(feature_ood, w, b)

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")

    # ---------------------------------------
    method = "Mahalanobis"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    print("\tcomputing classwise mean feature...")
    train_means = []
    train_feat_centered = []
    for i in tqdm(range(1000)):
        fs = feature_id_train[train_labels == i]
        _m = fs.mean(axis=0)
        train_means.append(_m)
        train_feat_centered.extend(fs - _m)

    print("\tcomputing precision matrix...")
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(train_feat_centered).astype(np.float64))

    print("\tgo to gpu...")
    mean = torch.from_numpy(np.array(train_means)).cuda().float()
    prec = torch.from_numpy(ec.precision_).cuda().float()

    score_id = -np.array(
        [
            (((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
            for f in tqdm(torch.from_numpy(feature_id_val).cuda().float())
        ]
    )
    for name, feature_ood in feature_oods.items():
        score_ood = -np.array(
            [
                (((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
                for f in tqdm(torch.from_numpy(feature_ood).cuda().float())
            ]
        )

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")

    # ---------------------------------------
    method = "KL-Matching"
    print(f"\n{method}")
    result = []
    fw.write(f"{method}\n")

    print("\tcomputing classwise mean softmax...")
    pred_labels_train = np.argmax(softmax_id_train, axis=-1)
    mean_softmax_train = [
        softmax_id_train[pred_labels_train == i].mean(axis=0) for i in tqdm(range(1000))
    ]

    score_id = -pairwise_distances_argmin_min(
        softmax_id_val, np.array(mean_softmax_train), metric=kl
    )[1]
    for name, softmax_ood in softmax_oods.items():
        score_ood = -pairwise_distances_argmin_min(
            softmax_ood, np.array(mean_softmax_train), metric=kl
        )[1]

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)

        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
        fw.write(f"\t{name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f"\tmean auroc {df.auroc.mean():.2%}, fpr {df.fpr.mean():.2%}")


if __name__ == "__main__":
    main()
