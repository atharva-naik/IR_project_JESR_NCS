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

**triplet contrastive pretraining (TCP):** <br>
Create triplet instances from the conala_mixed.jsonl using the following method: <br>
```for each post title``` <br>
1) sample positive examples (code snippets) from a given post. <br>
2) for each postive sample, sample (k=3) "negative" code snippets from other posts (post titles). <br>

**intra-category negative sampling (ICNS):** 
for each

**relevance thresholding for positive samples (RTPS):** 

|model name|recall@5|recall@10|mrr|avg_candidate_rank|avg_best_candidate_rank|
|---|---|---|---|---|---|
|CodeBERT_zero_shot|0.018|0.022|0.015|242.406|229.819|
|CodeBERT_zero_shot|0.012|0.022|0.018|232.462|217.389|
|CodeBERT_zero_shot|0.016|0.026|0.017|237.924|225.027|
|CodeBERT_zero_shot|0.030|0.050|0.028|223.428|204.353|
|CodeBERT_zero_shot|0.194|0.244|0.165|149.210|141.545|
|CodeBERT_zero_shot|0.086|0.124|0.076|183.616|168.751|
|triplet_CodeBERT|0.590|0.732|0.497|11.016|8.299|
|triplet_CodeBERT|0.764|0.850|0.676|15.336|13.932|
|triplet_CodeBERT|0.746|0.846|0.642|6.548|5.238|
|triplet_CodeBERT|0.622|0.780|0.519|8.342|6.115|
|triplet_CodeBERT|0.792|0.876|0.706|6.696|5.756|
|triplet_CodeBERT|0.800|0.882|0.685|4.454|3.584|
|triplet_CodeBERT_intra_categ_neg|0.586|0.730|0.505|9.570|7.068|
|triplet_CodeBERT_intra_categ_neg|0.774|0.852|0.675|16.452|14.279|
|triplet_CodeBERT_intra_categ_neg|0.726|0.834|0.625|6.602|4.775|
|triplet_CodeBERT_intra_categ_neg|0.632|0.792|0.535|7.612|5.501|
|triplet_CodeBERT_intra_categ_neg|0.772|0.860|0.690|7.540|6.184|
|triplet_CodeBERT_intra_categ_neg|0.768|0.878|0.670|4.836|3.529|
|triplet_CodeBERT_rel_thresh|0.602|0.740|0.511|10.478|8.499|
|triplet_CodeBERT_rel_thresh|0.786|0.858|0.685|18.990|16.107|
|triplet_CodeBERT_rel_thresh|0.754|0.858|0.644|7.078|5.647|
|triplet_CodeBERT_rel_thresh|0.642|0.766|0.553|9.208|7.049|
|triplet_CodeBERT_rel_thresh|0.796|0.866|0.705|12.092|9.605|
|triplet_CodeBERT_rel_thresh|0.760|0.866|0.691|5.130|4.071|
|triplet_CodeBERT_rel_thresh_intra_categ_neg|0.562|0.690|0.475|13.548|9.586|
|triplet_CodeBERT_rel_thresh_intra_categ_neg|0.782|0.854|0.662|19.726|17.310|
|triplet_CodeBERT_rel_thresh_intra_categ_neg|0.722|0.820|0.614|8.638|6.671|
|triplet_CodeBERT_rel_thresh_intra_categ_neg|0.606|0.718|0.498|11.196|7.707|
|triplet_CodeBERT_rel_thresh_intra_categ_neg|0.808|0.862|0.679|11.744|10.173|
|triplet_CodeBERT_rel_thresh_intra_categ_neg|0.770|0.870|0.668|5.868|4.425|

<!-- |model name|recall@5|recall@10|mrr|avg_candidate_rank|avg_best_candidate_rank|
|---|---|---|---|---|---|
|CodeBERT (zero shot)|0.030|0.050|0.028|223.428|204.353|
|CodeBERT |0.622|0.780|0.519|8.342|6.115|
|CodeBERT (ICNS)|0.632|**0.792**|0.535|**7.612**|**5.501**|
|CodeBERT (RTPS)|**0.642**|0.766|**0.553**|9.208|7.049|
|CodeBERT (ICNS + RTPS)|0.606|0.718|0.498|11.196|7.707| -->

<!-- |model name|recall@5|recall@10|avg_candidate_rank|avg_best_candidate_rank|mrr|
|---|---|---|---|---|---|
|CodeBERT_zero_shot|0.030|0.050|223.428|204.353|0.028|
|triplet_CodeBERT|0.622|0.780|8.342|6.115|0.519|
|triplet_CodeBERT_intra_categ_neg|0.632|**0.792**|**7.612**|**5.501**|0.535|
|triplet_CodeBERT_rel_thresh|**0.642**|0.766|9.208|7.049|**0.553**|
|triplet_CodeBERT_rel_thresh_intra_categ_neg|0.606|0.718|11.196|7.707|0.498|

|model name|recall@5|recall@10|avg_candidate_rank|avg_best_candidate_rank|mrr|
|---|---|---|---|---|---|
|CodeBERT_zero_shot|0.030|0.050|223.428|204.353|0.028|
|triplet_CodeBERT|0.622|0.780|8.342|6.115|0.519|
|triplet_CodeBERT_intra_categ_neg|0.632|0.792|7.612|5.501|0.535|
|triplet_CodeBERT_rel_thresh|0.642|0.766|9.208|7.049|0.553|
|triplet_CodeBERT_rel_thresh_intra_categ_neg|0.592|0.736|10.400|7.036|0.492|

|model name|recall@5|recall@10|avg_candidate_rank|avg_best_candidate_rank|mrr|
|---|---|---|---|---|---|
|CodeBERT_zero_shot|0.030|0.050|223.428|204.353|0.028|
|triplet_CodeBERT|0.622|0.780|8.342|6.115|0.519|
|triplet_CodeBERT_intra_categ_neg|0.622|**0.788**|**7.558**|**5.162**|0.542|
|triplet_CodeBERT_rel_thresh|**0.642**|0.766|9.208|7.049|**0.553**|
|triplet_CodeBERT_rel_thresh_intra_categ_neg|0.592|0.736|10.400|7.036|0.492|

|model name|recall@5|recall@10|avg_candidate_rank|avg_best_candidate_rank|mrr|
|---|---|---|---|---|---|
|CodeBERT_zero_shot|0.030|0.050|223.428|204.353|0.028|
|triplet_CodeBERT|0.622|0.780|8.342|6.115|0.519|
|triplet_CodeBERT_intra_categ_neg|0.666|0.774|7.782|5.436|0.548|
|triplet_CodeBERT_rel_thresh|0.642|0.766|9.208|7.049|0.553|
|triplet_CodeBERT_rel_thresh_intra_categ_neg|0.572|0.734|10.050|6.910|0.475| -->