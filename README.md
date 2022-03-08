# IR_project_JESR_NCS
Implementation of Joint Embedding Space Retrieval (JESR) for Neural Code Search (NCS) for IR term project.

Our model is composed of the following components (1) encoders (2) distance function
## Available Encoders
We use different encoders for the text and code

### Text Encoders
1. BERT
2. RoBERTa
3. Sentence-BERT

### Code Encoders
1. Code2Vec
2. Roberta (code)

### Performance Comparison
avg canditate rank: 8.342
avg best candidate rank: 6.115068493150685
MRR (LRAP): 0.5192234969909357

avg canditate rank: 11.016
avg best candidate rank: 8.2986301369863
MRR (LRAP): 0.49738491711917343

avg canditate rank: 223.428
avg best candidate rank: 204.35342465753425
MRR (LRAP): 0.02759542885829559

|model name|recall@5|recall@10|mrr|avg_candidate_rank|avg_best_candidate_rank|
|---|---|---|---|---|---|
|CodeBERT_zero_shot|0.030|0.050|0.028|223.428|204.353|
|triplet_CodeBERT|0.622|0.780|0.519|8.342|6.115|
|triplet_CodeBERT_intra_categ_neg|0.632|**0.792**|0.535|**7.612**|**5.501**|
|triplet_CodeBERT_rel_thresh|**0.642**|0.766|**0.553**|9.208|7.049|
|triplet_CodeBERT_rel_thresh_intra_categ_neg|0.606|0.718|0.498|11.196|7.707|

|model name|recall@5|recall@10|avg_candidate_rank|avg_best_candidate_rank|mrr|
|---|---|---|---|---|---|
|CodeBERT_zero_shot|0.030|0.050|223.428|204.353|0.028|
|triplet_CodeBERT|0.622|0.780|8.342|6.115|0.519|
|triplet_CodeBERT_intra_categ_neg|0.632|**0.792**|**7.612**|**5.501**|0.535|
|triplet_CodeBERT_rel_thresh|**0.642**|0.766|9.208|7.049|**0.553**|
|triplet_CodeBERT_rel_thresh_intra_categ_neg|0.606|0.718|11.196|7.707|0.498|

<!-- |model name|recall@5|recall@10|avg_candidate_rank|avg_best_candidate_rank|mrr|
|---|---|---|---|---|---|
|CodeBERT_zero_shot|0.030|0.050|223.428|204.353|0.028|
|triplet_CodeBERT|0.622|0.780|8.342|6.115|0.519|
|triplet_CodeBERT_intra_categ_neg|0.632|0.792|7.612|5.501|0.535|
|triplet_CodeBERT_rel_thresh|0.642|0.766|9.208|7.049|0.553|
|triplet_CodeBERT_rel_thresh_intra_categ_neg|0.592|0.736|10.400|7.036|0.492| -->
<!-- |model name|recall@5|recall@10|avg_candidate_rank|avg_best_candidate_rank|mrr|
|---|---|---|---|---|---|
|CodeBERT_zero_shot|0.030|0.050|223.428|204.353|0.028|
|triplet_CodeBERT|0.622|0.780|8.342|6.115|0.519|
|triplet_CodeBERT_intra_categ_neg|0.622|**0.788**|**7.558**|**5.162**|0.542|
|triplet_CodeBERT_rel_thresh|**0.642**|0.766|9.208|7.049|**0.553**|
|triplet_CodeBERT_rel_thresh_intra_categ_neg|0.592|0.736|10.400|7.036|0.492| -->
<!--  |model name|recall@5|recall@10|avg_candidate_rank|avg_best_candidate_rank|mrr|
|---|---|---|---|---|---|
|CodeBERT_zero_shot|0.030|0.050|223.428|204.353|0.028|
|triplet_CodeBERT|0.622|0.780|8.342|6.115|0.519|
|triplet_CodeBERT_intra_categ_neg|0.666|0.774|7.782|5.436|0.548|
|triplet_CodeBERT_rel_thresh|0.642|0.766|9.208|7.049|0.553|
|triplet_CodeBERT_rel_thresh_intra_categ_neg|0.572|0.734|10.050|6.910|0.475| -->