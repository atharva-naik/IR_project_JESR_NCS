python -m models.UniXcoder -t -tp data/conala-mined-100k_train.json -vp data/conala-mined-100k_val.json -en UniXcoder_intent_dyn_neg_sample_100k -d "cuda:0" -bs 48 -idns -e 3 -sip "CoNaLa_top10_sim_intents.json"