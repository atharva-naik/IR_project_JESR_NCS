#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import pprint
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib documentation:
def heatmap(data, row_labels, col_labels, ax=None,
            add_colorbar: bool=False, cbar_kw={}, 
            cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if not ax: ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # Create colorbar
    cbar = None
    if add_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, 
             ha="right", rotation_mode="anchor")
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.
    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)
    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

DATASET_NAMES = ["CoNaLa", "PyDocs"]
MODELS = ["UniXcoder", "CodeBERT", "GraphCodeBERT"]
exp_paths = {model: [] for model in MODELS}
EXP_NAMES = ["CNL (train)", "CNL (train) DNS", "PD (train)"]
for model in MODELS:
    for dataset_name, split in zip(["CoNaLa", "CoNaLa", "PyDocs"], ["_100k", "_dyn_neg_sample_100k", "_external_knowledge"]):
        exp_path = os.path.join("experiments", model+split)
        exp_paths[model].append((dataset_name, exp_path))
# add the baselines.
BASELINES = ["CNN", "RNN", "nBOW"]
for baseline in BASELINES:
    exp_paths[baseline] = [(
        "CoNaLa", os.path.join(
            "experiments", 
            f"{baseline.lower()}_siamese",
        ),
    )]
# print(exp_paths)
# metric names.
METRICS = ["mrr", "ndcg", "recall@5", "recall@10"]
TESTSET_NAMES = ["CoNaLa", "PyDocs", "Web Query", "CodeSearchNet"]
R = len(EXP_NAMES)
C = len(TESTSET_NAMES)
model_metric_grids = {
    "CodeBERT": {name: np.zeros((R,C)).tolist() for name in METRICS},
    "UniXcoder": {name: np.zeros((R,C-1)).tolist() for name in METRICS},
    "GraphCodeBERT": {name: np.zeros((R,C)).tolist() for name in METRICS},
    "CNN": {name: np.zeros((R-2,C)).tolist() for name in METRICS},
    "RNN": {name: np.zeros((R-2,C)).tolist() for name in METRICS},
    "nBOW": {name: np.zeros((R-2,C)).tolist() for name in METRICS},
}
for model, paths in exp_paths.items():
    print(paths)
    i = 0
    for train_dataset_name, exp_folder in paths:
        ood_path = os.path.join(exp_folder, "ood_test_metrics_l2_code.json")
        if not os.path.exists(ood_path): continue
        ood_data = json.load(open(ood_path))
        for j, test_dataset in enumerate(TESTSET_NAMES):
            if model == "UniXcoder" and test_dataset == "CodeSearchNet": continue
            test_datakey = {
                "CoNaLa": "CoNaLa", 
                "Web Query": "Web Query",
                "PyDocs": "External Knowledge",
                "CodeSearchNet": "CodeSearchNet",
            }[test_dataset]
            model_metric_grids[model]["mrr"][i][j] = ood_data[test_datakey]["mrr"]
            model_metric_grids[model]["ndcg"][i][j] = ood_data[test_datakey]["ndcg"]
            model_metric_grids[model]["recall@5"][i][j] = ood_data[test_datakey]["recall"]["@5"]
            model_metric_grids[model]["recall@10"][i][j] = ood_data[test_datakey]["recall"]["@10"]
        i += 1
pprint.pprint(model_metric_grids)
abbreviation = {
    "CoNaLa": "CNL",
    "PyDocs": "PD",
    "Web Query": "WQ",
    "CodeSearchNet": "CSN",
}
for model in model_metric_grids:
    # clear previous plots.
    plt.clf()
    # for ax in axs.flat:
    #     ax.set(xlabel='x-label', ylabel='y-label')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()
    fig, axs = plt.subplots(2, 2)
    for k, name in enumerate(model_metric_grids[model]):
        j = k % 2
        i = k // 2
        if model in BASELINES: 
            row_names = ["CNL (train)"]
        else: row_names = EXP_NAMES
        if model == "UniXcoder":
            col_names = [f"{abbreviation[dataset]} (test)" for dataset in TESTSET_NAMES[:-1]]
        else: col_names = [f"{abbreviation[dataset]} (test)" for dataset in TESTSET_NAMES]
        data = 100*np.array(model_metric_grids[model][name])
        im, cbar = heatmap(data, row_names, col_names, 
                           ax=axs[i, j], cmap="YlGn")
        texts = annotate_heatmap(im, valfmt="{x:.2f}", fontsize=10)
        axs[i, j].set_title(name)
    fig.tight_layout()
    fig.suptitle(f'{model} OOD Generalization Metrics\n', x=0.5, y=1)
    # va="bottom", fontsize=14)
    plt.savefig(f"plots/{model}_ood_metrics_grid.png")