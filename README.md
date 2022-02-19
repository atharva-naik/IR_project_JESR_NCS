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