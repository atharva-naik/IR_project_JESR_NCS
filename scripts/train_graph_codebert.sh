python -m models.GraphCodeBERT -t -tp triples/triples_rel_thresh_intra_categ_neg_train.json -vp triples/triples_rel_thresh_intra_categ_neg_val.json -en GraphCodeBERT_experiment --device_id "cuda:0" -bs 32