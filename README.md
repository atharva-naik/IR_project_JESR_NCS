# IR_project_JESR_NCS
Implementation of Joint Embedding Space Retrieval (JESR) for Neural Code Search (NCS) for IR term project.
Our model is composed of the following components (1) encoders (2) distance function


## Triplet Generation Process
**triplet contrastive pretraining (TCP):** <br>
Create triplet instances from the conala_mixed.jsonl using the following method: <br>
```for each post title``` <br>
1) sample positive examples (code snippets) from a given post. <br>
2) for each postive sample, sample (k=3) "negative" code snippets from other posts (post titles). <br>

**intra-category negative sampling (ICNS):** 
for each

## Baselines
**NOTE: Please grant execution permissions to all bash scripts (chmod +x)**

We train our baselines on **natural language** (nl) and **code snippet** (pl or programming language) pair classification task. 
We create a balanced training and validation set by sampling positive and negative instances from the CoNaLa mined pairs dataset.
We utilize a separate encoder for both text and code, and train the models in a siamese configuration, with Binary Cross Entropy as loss.
We encode code, text and annotations separately durin test time and score them using functions like inner product and euclidean distance (l2_loss)

### n-BOW: Neural Bag of words
1. Treat **nl** & **pl** as bag of words and represent them as mean pool of token level embeddings.
2. Utilize tokenizer of CodeBERT to get token sequence and initialize embedding layer with CodeBERT embeddings (768 dim).

To train model from scratch:
``` scripts/train_nbow.sh ```

To test saved model:
``` scripts/predict_nbow.sh ```

### CNN: Convolutional Neural Network
1. Perform 1-D convolutions with 3 filters of kernel width of 16 each and residual connections.
2. Use self-attention like weighted sum layer to pool the sequence output (across sequence lenght dim.)
3. Utilize tokenizer of CodeBERT to get token sequence but initialize embedding layer from scratch (128 dim.). We initialize from scratch unlike other baselines because we had performance issues on initializing with CodeBERT embeddings. 

To train model from scratch:
``` scripts/train_cnn.sh ```

To test saved model:
``` scripts/predict_cnn.sh ```

### RNN: Recurrent Neural Network (LSTM)
1. Treat **nl** & **pl** 
2. Utilize tokenizer of CodeBERT to get token sequence and initialize embedding layer with CodeBERT embeddings (768 dim).

To train model from scratch:
``` scripts/train_rnn.sh ```

To test saved model:
``` scripts/predict_rnn.sh ```

## Models

### Performance Comparison
**relevance thresholding for positive samples (RTPS):** 
|model name|recall@5|recall@10|mrr|avg_candidate_rank|avg_best_candidate_rank|ndcg|
|---|---|---|---|---|---|---|
|zero_shot (inner_prod) (code)|0.018|0.022|0.015|242.406|229.819|0.152|
|zero_shot (inner_prod) (annot)|0.012|0.022|0.018|232.462|217.389|0.156|
|zero_shot (inner_prod) (code+annot)|0.016|0.026|0.017|237.924|225.027|0.155|
|zero_shot (l2_dist) (code)|0.030|0.050|0.028|223.428|204.353|0.167|
|**zero_shot (l2_dist) (annot)**|**0.194**|**0.244**|**0.165**|**149.210**|**141.545**|**0.299**|
|zero_shot (l2_dist) (code+annot)|0.086|0.124|0.076|183.616|168.751|0.218|
|CodeBERT (inner_prod) (code)|0.590|0.732|0.497|11.016|8.299|0.615|
|CodeBERT (inner_prod) (annot)|0.764|0.850|0.676|15.336|13.932|0.749|
|CodeBERT (inner_prod) (code+annot)|0.746|0.846|0.642|6.548|5.238|0.726|
|CodeBERT (l2_dist) (code)|0.622|0.780|0.519|8.342|6.115|0.634|
|**CodeBERT (l2_dist) (annot)**|**0.792**|**0.876**|**0.706**|**6.696**|**5.756**|**0.774**|
|CodeBERT (l2_dist) (code+annot)|0.800|0.882|0.685|4.454|3.584|0.762|
|intra_categ_neg (inner_prod) (code)|0.586|0.730|0.505|9.570|7.068|0.621|
|intra_categ_neg (inner_prod) (annot)|0.774|0.852|0.675|16.452|14.279|0.751|
|intra_categ_neg (inner_prod) (code+annot)|0.726|0.834|0.625|6.602|4.775|0.714|
|intra_categ_neg (l2_dist) (code)|0.632|0.792|0.535|7.612|5.501|0.646|
|**intra_categ_neg (l2_dist) (annot)**|**0.772**|**0.860**|**0.690**|**7.540**|**6.184**|**0.764**|
|intra_categ_neg (l2_dist) (code+annot)|0.768|0.878|0.670|4.836|3.529|0.751|
|rel_thresh (inner_prod) (code)|0.602|0.740|0.511|10.478|8.499|0.625|
|rel_thresh (inner_prod) (annot)|0.786|0.858|0.685|18.990|16.107|0.757|
|rel_thresh (inner_prod) (code+annot)|0.754|0.858|0.644|7.078|5.647|0.729|
|rel_thresh (l2_dist) (code)|0.642|0.766|0.553|9.208|7.049|0.658|
|**rel_thresh (l2_dist) (annot)**|**0.796**|**0.866**|**0.705**|**12.092**|**9.605**|**0.773**|
|rel_thresh (l2_dist) (code+annot)|0.760|0.866|0.691|5.130|4.071|0.766|
|rel_thresh_intra_categ_neg (inner_prod) (code)|0.562|0.690|0.475|13.548|9.586|0.595|
|rel_thresh_intra_categ_neg (inner_prod) (annot)|0.782|0.854|0.662|19.726|17.310|0.739|
|rel_thresh_intra_categ_neg (inner_prod) (code+annot)|0.722|0.820|0.614|8.638|6.671|0.706|
|rel_thresh_intra_categ_neg (l2_dist) (code)|0.606|0.718|0.498|11.196|7.707|0.615|
|**rel_thresh_intra_categ_neg (l2_dist) (annot)**|**0.808**|**0.862**|**0.679**|**11.744**|**10.173**|**0.753**|
|rel_thresh_intra_categ_neg (l2_dist) (code+annot)|0.770|0.870|0.668|5.868|4.425|0.747|
|Graph zero_shot (inner_prod) (code)|0.132|0.190|0.096|187.690|175.052|0.238|
|**Graph zero_shot (inner_prod) (annot)**|**0.308**|**0.416**|**0.219**|**77.658**|**74.995**|**0.361**|
|Graph zero_shot (inner_prod) (code+annot)|0.216|0.276|0.176|136.228|127.452|0.313|
|Graph zero_shot (l2_dist) (code)|0.120|0.172|0.099|202.382|190.548|0.237|
|Graph zero_shot (l2_dist) (annot)|0.284|0.336|0.217|104.924|95.784|0.353|
|Graph zero_shot (l2_dist) (code+annot)|0.246|0.286|0.183|168.970|158.405|0.313|
|Graph CodeBERT (inner_prod) (code)|0.608|0.746|0.542|10.500|8.647|0.649|
|Graph CodeBERT (inner_prod) (annot)|0.800|0.874|0.712|15.074|13.244|0.779|
|Graph CodeBERT (inner_prod) (code+annot)|0.762|0.850|0.680|6.638|5.033|0.756|
|Graph CodeBERT (l2_dist) (code)|0.662|0.792|0.570|8.636|7.047|0.673|
|Graph CodeBERT (l2_dist) (annot)|0.822|0.886|0.713|6.560|5.416|0.781|
|**Graph CodeBERT (l2_dist) (code+annot)**|**0.818**|**0.896**|**0.717**|**4.474**|**3.490**|**0.786**|
|Graphintra_categ_neg (inner_prod) (code)|0.650|0.770|0.534|9.720|7.041|0.645|
|Graph intra_categ_neg (inner_prod) (annot)|0.778|0.862|0.678|17.406|15.490|0.753|
|Graph intra_categ_neg (inner_prod) (code+annot)|0.754|0.854|0.652|7.004|5.156|0.735|
|Graph intra_categ_neg (l2_dist) (code)|0.684|0.802|0.574|8.088|5.819|0.677|
|**Graph intra_categ_neg (l2_dist) (annot)**|**0.796**|**0.870**|**0.703**|**8.102**|**6.781**|**0.773**|
|Graph intra_categ_neg (l2_dist) (code+annot)|0.788|0.898|0.695|4.910|3.562|0.770|
|Graph rel_thresh (inner_prod) (code)|0.662|0.772|0.557|9.152|6.704|0.660|
|Graph rel_thresh (inner_prod) (annot)|0.794|0.854|0.700|16.422|13.419|0.768|
|Graph rel_thresh (inner_prod) (code+annot)|0.782|0.864|0.686|6.596|4.636|0.761|
|Graph rel_thresh (l2_dist) (code)|0.670|0.788|0.574|8.128|5.323|0.676|
|**Graph rel_thresh (l2_dist) (annot)**|**0.808**|**0.878**|**0.723**|**8.572**|**6.803**|**0.788**|
|Graph rel_thresh (l2_dist) (code+annot)|0.816|0.884|0.714|4.246|2.942|0.784|
|Graph rel_thresh_intra_categ_neg (inner_prod) (code)|0.598|0.738|0.521|11.112|8.789|0.633|
|Graph rel_thresh_intra_categ_neg (inner_prod) (annot)|0.772|0.854|0.667|16.068|13.225|0.743|
|Graph rel_thresh_intra_categ_neg (inner_prod) (code+annot)|0.748|0.852|0.636|7.016|5.384|0.723|
|Graph rel_thresh_intra_categ_neg (l2_dist) (code)|0.634|0.758|0.540|9.690|7.397|0.649|
|**Graph rel_thresh_intra_categ_neg (l2_dist) (annot)**|**0.792**|**0.868**|**0.709**|**9.444**|**7.501**|**0.777**|
|Graph rel_thresh_intra_categ_neg (l2_dist) (code+annot)|0.780|0.882|0.679|5.076|3.907|0.758|
|nbow_siamese (inner_prod) (code)|0.268|0.382|0.197|35.456|26.071|0.360|
|nbow_siamese (inner_prod) (annot)|0.248|0.362|0.186|55.050|45.625|0.344|
|nbow_siamese (inner_prod) (code+annot)|0.286|0.414|0.221|33.052|24.490|0.380|
|nbow_siamese (l2_dist) (code)|0.080|0.096|0.062|154.880|143.016|0.208|
|**nbow_siamese (l2_dist) (annot)**|**0.406**|**0.492**|**0.309**|**63.392**|**59.537**|**0.440**|
|nbow_siamese (l2_dist) (code+annot)|0.202|0.278|0.138|94.118|82.625|0.290|
|cnn_siamese (inner_prod) (code)|0.100|0.182|0.087|69.922|56.085|0.252|
|cnn_siamese (inner_prod) (annot)|0.086|0.142|0.081|87.168|75.748|0.240|
|cnn_siamese (inner_prod) (code+annot)|0.120|0.198|0.104|64.360|53.016|0.269|
|cnn_siamese (l2_dist) (code)|0.054|0.104|0.060|102.886|84.121|0.219|
|**cnn_siamese (l2_dist) (annot)**|**0.198**|**0.288**|**0.153**|**79.670**|**68.995**|**0.306**|
|cnn_siamese (l2_dist) (code+annot)|0.166|0.266|0.133|74.488|63.082|0.295|
|rnn_siamese (inner_prod) (code)|0.224|0.334|0.182|46.534|39.060|0.343|
|rnn_siamese (inner_prod) (annot)|0.412|0.510|0.317|41.112|37.890|0.456|
|rnn_siamese (inner_prod) (code+annot)|0.410|0.540|0.315|29.556|26.885|0.458|
|rnn_siamese (l2_dist) (code)|0.172|0.272|0.144|62.688|53.707|0.304|
|rnn_siamese (l2_dist) (annot)|0.464|0.546|0.369|36.758|34.063|0.498|
|**rnn_siamese (l2_dist) (code+annot)**|**0.474**|**0.590**|**0.396**|**27.850**|**26.408**|**0.523**|

<!-- 
|model name|recall@5|recall@10|mrr|avg_candidate_rank|avg_best_candidate_rank|ndcg|
|---|---|---|---|---|---|---|
|zero_shot (inner_prod) (code)|0.018|0.022|0.015|242.406|229.819|0.152|
|zero_shot (inner_prod) (annot)|0.012|0.022|0.018|232.462|217.389|0.156|
|zero_shot (inner_prod) (code+annot)|0.016|0.026|0.017|237.924|225.027|0.155|
|zero_shot (l2_dist) (code)|0.030|0.050|0.028|223.428|204.353|0.167|
|zero_shot (l2_dist) (annot)|0.194|0.244|0.165|149.210|141.545|0.299|
|zero_shot (l2_dist) (code+annot)|0.086|0.124|0.076|183.616|168.751|0.218|
|CodeBERT (inner_prod) (code)|0.590|0.732|0.497|11.016|8.299|0.615|
|CodeBERT (inner_prod) (annot)|0.764|0.850|0.676|15.336|13.932|0.749|
|CodeBERT (inner_prod) (code+annot)|0.746|0.846|0.642|6.548|5.238|0.726|
|CodeBERT (l2_dist) (code)|0.622|0.780|0.519|8.342|6.115|0.634|
|CodeBERT (l2_dist) (annot)|0.792|0.876|0.706|6.696|5.756|0.774|
|CodeBERT (l2_dist) (code+annot)|0.800|0.882|0.685|4.454|3.584|0.762|
|intra_categ_neg (inner_prod) (code)|0.586|0.730|0.505|9.570|7.068|0.621|
|intra_categ_neg (inner_prod) (annot)|0.774|0.852|0.675|16.452|14.279|0.751|
|intra_categ_neg (inner_prod) (code+annot)|0.726|0.834|0.625|6.602|4.775|0.714|
|intra_categ_neg (l2_dist) (code)|0.632|0.792|0.535|7.612|5.501|0.646|
|intra_categ_neg (l2_dist) (annot)|0.772|0.860|0.690|7.540|6.184|0.764|
|intra_categ_neg (l2_dist) (code+annot)|0.768|0.878|0.670|4.836|3.529|0.751|
|rel_thresh (inner_prod) (code)|0.602|0.740|0.511|10.478|8.499|0.625|
|rel_thresh (inner_prod) (annot)|0.786|0.858|0.685|18.990|16.107|0.757|
|rel_thresh (inner_prod) (code+annot)|0.754|0.858|0.644|7.078|5.647|0.729|
|rel_thresh (l2_dist) (code)|0.642|0.766|0.553|9.208|7.049|0.658|
|rel_thresh (l2_dist) (annot)|0.796|0.866|0.705|12.092|9.605|0.773|
|rel_thresh (l2_dist) (code+annot)|0.760|0.866|0.691|5.130|4.071|0.766|
|rel_thresh_intra_categ_neg (inner_prod) (code)|0.562|0.690|0.475|13.548|9.586|0.595|
|rel_thresh_intra_categ_neg (inner_prod) (annot)|0.782|0.854|0.662|19.726|17.310|0.739|
|rel_thresh_intra_categ_neg (inner_prod) (code+annot)|0.722|0.820|0.614|8.638|6.671|0.706|
|rel_thresh_intra_categ_neg (l2_dist) (code)|0.606|0.718|0.498|11.196|7.707|0.615|
|rel_thresh_intra_categ_neg (l2_dist) (annot)|0.808|0.862|0.679|11.744|10.173|0.753|
|rel_thresh_intra_categ_neg (l2_dist) (code+annot)|0.770|0.870|0.668|5.868|4.425|0.747| -->

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