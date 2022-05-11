# Intent Categories
Categorization of queries based on whether they are looking for snippets that are **data oriented** or perform small **procedures**.

## Procedural:
data type conversion, computing values (reduction) from data structure, splitting string (string -> list) data type conversion

## Data Oriented:
How to identify: only a data structure is modified without changing the data type.
Operations included: 
1) create, initialization (creating multidim array) <br>
2) sorting <br>
3) adding <br>
4) removing <br>
5) updating values <br>
6) reversing, <br> 
7) find position of item <br> 
8) element wise transformation <br> 
9) max <br>
10) min <br> 
11) append <br> 
12) slicing <br>
13) splitting <br>
14) concatenating <br>
15) access, get <br>

### Unresolved Questions: 
How should rounding be categorized?

### Slightly Ambiguous instances:
219	Python - how to round down to 2 decimals <br>
98	Coalesce values from 2 columns into a single column in a pandas dataframe <br>
125	Merging a list of lists <br>

# Error Categorization Pipeline
We are designing this pipeline in a way that could allows to reuse it in our pre-training process, for generating triplets belonging to particular error classes. <br>
We assume that the input to the pipeline is a triplet containing:
1) Intent/Query <br>
2) Code Snippet (+ve example) <br>
3) Retrieved Snippet (-ve example for errors) <br>

## Components:
The main components of our pipeline.

### Intent Classifier (IC): 
Classify intents as data centric or procedural (input = intent). Naive Bayes should work. Create a small human annotated test set for evaluation.

### DataType Extractor (DE):
Use parsers to identify data type using positive snippet and intent, or negative snippet. Data type slot classifier: identifies data structure being operated on (input = intent (optional), snippet)

### OpType Extractor (OE):
Extract operator being used from code snippet or code snippet and intent given a particular datatype (input = intent (optional), snippet, datatype)

### Procedure Name Extractor (PNE):
Procedure name identifier: identifies name of function that is being operation (input = snippet). Use the ast library

### Arguments Extractor (AE):
Extract argument list of a given function from the code snippet Use the ast library. (input=snippet, procedure)

## Algorithm:
Error categorization steps:
