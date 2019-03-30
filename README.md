# SemEval-2018 Task 7 Subtask 1 - Relation Classification

A PKU course project based on the "SemEval-2018 task 7 Semantic Relation Extraction and Classification in Scientific Papers" dataset.

The Subtask 1

- 1.1 Relation classification on clean data
- 1.2 Relation classification on noisy data

## Table of Content

- [SemEval-2018 Task 7 Subtask 1 - Relation Classification](#semeval-2018-task-7-subtask-1---relation-classification)
  - [Table of Content](#table-of-content)
  - [Competition](#competition)
    - [Semantic Relations](#semantic-relations)
    - [Subtask 1: Relation classification](#subtask-1-relation-classification)
      - [1.1: Relation classification on clean data](#11-relation-classification-on-clean-data)
      - [1.2: Relation classification on noisy data](#12-relation-classification-on-noisy-data)
    - [Evaluation](#evaluation)
  - [Using Traditional ML Method](#using-traditional-ml-method)
    - [Procedure](#procedure)
    - [The Idea of sentence embedding](#the-idea-of-sentence-embedding)
  - [Using Pure Embedding Model](#using-pure-embedding-model)
    - [Another way of sentence embedding](#another-way-of-sentence-embedding)
  - [Using LightRel as Baseline - Forked Project](#using-lightrel-as-baseline---forked-project)
    - [Case Study](#case-study)
      - [The Word Embedding](#the-word-embedding)
      - [The Cluster-membership](#the-cluster-membership)
      - [Feature](#feature)
      - [External Resource used](#external-resource-used)
      - [Conclusion](#conclusion)
    - [Test original performance](#test-original-performance)
    - [Modifications](#modifications)
      - [Try different embedding](#try-different-embedding)
      - [Add more feature](#add-more-feature)
  - [Using Deep Learning](#using-deep-learning)
    - [Learning from the extracted feature](#learning-from-the-extracted-feature)
    - [Learning from the beginning](#learning-from-the-beginning)
  - [Appendix: The performance of random](#appendix-the-performance-of-random)
  - [Trouble Shooting](#trouble-shooting)
    - [Execute shell command and get stdout](#execute-shell-command-and-get-stdout)
    - [XML](#xml)
    - [JSON](#json)
    - [#include <malloc.h> not found in OSX](#include-malloch-not-found-in-osx)
    - [Python Type hint](#python-type-hint)
    - [Get BERT word embedding](#get-bert-word-embedding)
    - [Random stuff](#random-stuff)
  - [TODO](#todo)
  - [Links](#links)
    - [TextCNN](#textcnn)
    - [Data](#data)
    - [Paper](#paper)

## Competition

### Semantic Relations

Relation instances are to be classified into one of the following relations: USAGE, RESULT, MODEL, PART_WHOLE, TOPIC, COMPARISON.

1. USAGE is an asymmetrical relation. It holds between two entities X and Y, where, for example:

   ```txt
   X is used for Y
   X is a method used to perform a task Y
   X is a tool used to process data Y
   X is a type of information/representation of information used by/in a system Y)
   ```

2. RESULT is an asymmetrical relation. It holds between two entities X and Y, where, for example:

   ```txt
   X gives as a result Y (where Y is typically a measure of evaluation)
   X yields Y (where Y is an improvement or decrease)
   a feature of a system or a phenomenon X yields Y (where Y is an improvement or decrease)
   ```

3. MODEL-FEATURE is an asymmetrical relation. It holds between two entities X and Y, where, for example:

   ```txt
   X is a feature/an observed characteristic of Y
   X is a model of Y
   X is a tag(set) used to represent Y
   ```

4. PART_WHOLE is an asymmetrical relation. It holds between two entities X and Y, where, for example:

   ```txt
   X is a part, a component of Y
   X is found in Y
   Y is built from/composed of X
   ```

5. TOPIC is an asymmetrical relation. It holds between two entities X and Y, where, for example:

   ```txt
   X deals with topic Y
   X (author, paper) puts forward Y (an idea, an approach)
   ```

6. COMPARE is a symmetrical relation. It holds between two entities X and Y, where:

   ```txt
   X is compared to Y (e.g. two systems, two feature sets or two results)
   ```

The counts for each relation:

- `USAGE`: 483
- `TOPIC`: 18
- `RESULT`: 72
- `PART_WHOLE`: 234
- `MODEL-FEATURE`: 326
- `COMPARE`: 95

In test set:

- `USAGE`: 175
- `TOPIC`: 3
- `RESULT`: 20
- `PART_WHOLE`: 70
- `MODEL-FEATURE`: 66
- `COMPARE`: 21

### Subtask 1: Relation classification

> For each subtask, training and test data include abstracts of papers from the ACL Anthology Corpus with pre-annotated entities that represent concepts. Two types of tasks are proposed:
>
> 1. identifying pairs of entities that are instances of any of the six semantic relations (extraction task),
>
> 2. classifying instances into one of the six semantic relation types (classification task).

The subtask is decomposed into two scenarios according to the data used: classification on clean data and classification on noisy data. The task is identical for both scenarios: given a relation instance consisting of two entities in context, predict the semantic relation between the entities. A relation instance is identified by the unique ID of the two entities.

For the subtask 1, instances with directionality are provided in both the training data and the test data and they are not to be modified or completed in the test data; the relation label is provided in the training data and has to be predicted for the test data.

#### 1.1: Relation classification on clean data

The classification task is performed on data where entities are manually annotated, following the ACL RD-TEC 2.0 guidelines. Entities represent domain concepts specific to NLP, while high-level scientific terms (e.g. "hypothesis", "experiment") are not annotated.

Example (annotated text):

```txt
<abstract>

The key features of the system include: (i) Robust efficient <entity id="H01-1041.8">parsing</entity> of <entity id="H01-1041.9">Korean</entity> (a <entity id="H01-1041.10">verb final language</entity> with <entity id="H01-1041.11">overt case markers</entity> , relatively <entity id="H01-1041.12">free word order</entity> , and frequent omissions of <entity id="H01-1041.13">arguments</entity> ). (ii) High quality <entity id="H01-1041.14">translation</entity> via <entity id="H01-1041.15">word sense disambiguation</entity> and accurate <entity id="H01-1041.16">word order generation</entity> of the <entity id="H01-1041.17">target language</entity> .(iii) <entity id="H01-1041.18">Rapid system development</entity> and porting to new <entity id="H01-1041.19">domains</entity> via <entity id="H01-1041.20">knowledge-based automated acquisition of grammars</entity> .

</abstract>
```

Relation instances in the annotated text (provided for test data):

```txt
(H01-1041.8,H01-1041.9)

(H01-1041.10,H01-1041.11,REVERSE)

(H01-1041.14,H01-1041.15,REVERSE)
```

Submission format with predictions:

```txt
USAGE(H01-1041.8, H01-1041.9)

MODEL-FEATURE(H01-1041.10, H01-1041.11,REVERSE)

USAGE(H01-1041.14, H01-1041.15,REVERSE)
```

#### 1.2: Relation classification on noisy data

The task is identical to 1.1., but the entities are annotated automatically and contain noise. The annotation comes from the ACL-RelAcS corpus and it is based on a combination of automatic terminology extraction and external ontologies1. Entities are therefore terms specific to the given corpus, and include high-level terms (e.g. "algorithm", "paper", "method"). They are not always full NPs and they may include noise (verbs, irrelevant words). Relations were manually annotated in the training data and in the gold standard, between automatically annotated entities. Do not try to correct entity annotation in any way in your submission.

Example (annotated text):

```txt
<abstract>

This <entity id="L08-1203.8">paper</entity> introduces a new <entity id="L08-1203.9">architecture</entity> that aims at combining molecular <entity id="L08-1203.10">biology</entity> <entity id="L08-1203.11">data</entity> with <entity id="L08-1203.12">information</entity> automatically <entity id="L08-1203.13">extracted</entity> from relevant <entity id="L08-1203.14">scientific literature</entity>

</abstract>
```

Relation instances in the annotated text (provided for test data):

```txt
(L08-1203.8,L08-1203.9)

(L08-1203.12,L08-1203.14)
```

Submission format for predictions:

```txt
TOPIC(L08-1203.8,L08-1203.9)

PART_WHOLE(L08-1203.12,L08-1203.14)
```

### Evaluation

For subtasks 1.1 and 1.2 which are usual classification tasks, the following class-based evaluation metrics are used:

- for every distinct class: precision, recall and F1-measure (β=1)
- global evaluation, for the set of classes:
  - macro-average of the F1-measures of every distinct class
  - micro-average of the F1-measures of every distinct class

---

Approach and Results

---

## Using Traditional ML Method

```sh
# Prepared the dataset and embedding
python3 text_processing.py

# Training and result
python3 ml_model.py
```

### Procedure

1. Load dataset
   1. Load the _relations_ into a hash map of `(entity 1, entity 2) -> relation`
   2. Load the _text_ into a Paper class object
      - paper id
      - title
      - abstract (plain text)
      - entity id and text
2. Sentence embedding => Get the training feature with label
3. Training model
   - SVM
   - Logistic Regression
4. Test the result
   - Using the SemEval 2018 task 7 scorer
   - Calculate by ourself (5-fold cross validation)

### The Idea of sentence embedding

Combine two entity with some relation words to form a sentence. And we use the sentence to classify whether it belong to this class.

Example:

```txt
USAGE(H01-1001.9,H01-1001.10)
<entity id="H01-1001.9">oral communication</entity>
<entity id="H01-1001.10">indices</entity>

USAGE(oral communication, indices) =>

oral communication used by indices
oral communication used for indices
oral communication applied to indices
oral communication performed on indices
```

We multiply each words' embedding and then normalize it. (or the value of result will be relevant to the sentence length)

But the result is not quite ideal.

## Using Pure Embedding Model

> TODO

### Another way of sentence embedding

Calculate three parts of embedding. `entity 1`, `relationship`, and `entity 2`.

> And two of them, doing dot product. Then we'll get 2 numbers. And use them to form the result.

## Using LightRel as Baseline - [Forked Project](https://github.com/pku-nlp-forfun/LightRel)

### Case Study

- LightRel trained the dictionary for SemEval 2010 Task 8 and reused it for SemEval 2010 Task 7

#### The Word Embedding

> [Pre-trained embedding by LightRel](https://cloud.dfki.de/owncloud/index.php/s/WKOCMj5UYiSVZeR)

LightRel used two different corpus from Citation Network Dataset.

Used only the _abstract_ part of `ACM-Citation-network V9` and `DBLP-Citation-network V5`

Because the abstract part is followed by `#! --- Abstract` thus use regular expression to extract it.

```sh
time grep -p "^#\!" acm.txt | sed 's/^#\!//' > acm_abstracts.txt
```

And trained the embedding using [word2vec](https://code.google.com/archive/p/word2vec/)

```sh
time word2vec -train abstracts-dblp-semeval2018.txt -output abstracts-dblp-semeval2018.wcs.txt -size 300 -min-count 5 -binary 0
```

- 300-dimension vector
- leaving out tokens occurring fewer than 5 times

#### The Cluster-membership

- [MarLiN - A fast word clustering tool](http://cistern.cis.lmu.de/marlin/)
  - [muelletm/cistern](https://github.com/muelletm/cistern)

#### Feature

- index a unique vocabulary immediately following the entity1 and immediately preceding entity two.

  ```txt
  ['information_retrieval_techniques','use','a','histogram','of','keywords']

  => 'use' and 'of'
  ```

- **word-shape feature**: a unique vector representing certain character-level features found in a word
  - any character is capitalized
  - a comma is present
  - the first character is capitalized and the word is the first in the relation
    - representing the beginning of a sentence
  - the first character is lower-case
  - there is an underscore present (representing a multi-word entity)
  - if quotes are present in the token

> These feature is continue using from SemEval 2010 Task 8

#### External Resource used

Addition library:

- [liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
- [ast](https://docs.python.org/3/library/ast.html) - Abstract Syntax Trees
- [unicodedata](https://docs.python.org/3/library/unicodedata.html)
- [svn2github/word2vec](https://github.com/svn2github/word2vec)
  - [dav/word2vec](https://github.com/dav/word2vec)
  - [original repository](https://code.google.com/archive/p/word2vec/))

Embedding corpus:

- [Citation Network Dataset](https://aminer.org/citation)
  - [dblp](https://dblp.uni-trier.de/)
  - [acm](https://www.acm.org/)

#### Conclusion

The model used by LightRel is not complicated. The performance is depend more on features (embedding).

### Test original performance

> These options can be on/off in `parameters.py`

**one-feature**:

other parameter: L2-regularized logistic regression, LR=1, epoch=0.1, cost=0.05

1. fire_word: 25.94%
2. fire_shape: 19.19%
3. **fire_embedding: 49.84%** (the most important feature)
4. fire_cluster: 27.41%
5. e1: 37.29%
6. e2: 31.92%
7. before_e2: 9.39%
8. null: 9.39%

**multiple-feature**:

1. embedding + e1 + e2: 49.90%

### Modifications

#### Try different embedding

> The [original embedding](#The-Word-Embedding) was created by word2vec

We use different corpus, `ACM-Citation-network V9` and `DBLP-Citation-network V10`

Extract the abstract from ACM using:

```sh
time grep -p "^#\!" acm.txt | sed -e '/^#\!First Page of the Article/d' | sed 's/^#\!//' >> abstracts.txt
```

And extract the abstract from DBLP with [jq](https://stedolan.github.io/jq/) using:

```bash
for file in `ls dblp-ref`
do
   time jq '.abstract' `dblp-ref/$file` | sed -e '/null/d' | sed 's/\"//g' >> abstracts.txt
done
```

or

```bash
for file in dblp-ref/*
do
   time jq '.abstract' $file | sed -e '/null/d' | sed 's/\"//g' >> abstracts.txt
done
```

We put all the content in same file called `abstracts.txt`.

```sh
# Done these things in one command
cd LightRel/embedding
bash corpusPreprocessing.sh
```

**Try word2vec**:

```sh
time word2vec -train abstracts.txt -output abstracts.wcs.txt -size 300 -min-count 5 -binary  0
```

**Try [BERT](https://github.com/google-research/bert)**:

using BERT-Large, Uncased model (24-layer, 1024-hidden, 16-heads, 340M parameters)

```sh
cd LightRel/embedding
bash trainBERT.sh
```

**Try [fastText](https://github.com/facebookresearch/fastText/)**:

```sh
cd LightRel/embedding
bash trainFastText.sh
```

**Performance**:

LibLinear (Version 2.30) LR: flods=5, cost=0.05, epoch=0.1

```sh
cd LightRel
bash lightRel.sh 5
```

| Embedding | Model        | Data              | Test F1 | Test P | Test R | Train F1 (%) | Train P | Train R | USAGE | TOPIC | RESULT | PART_WHOLE | MODEL-FEATURE | COMPARE |
| --------- | ------------ | ----------------- | ------- | ------ | ------ | ------------ | ------- | ------- | ----- | ----- | ------ | ---------- | ------------- | ------- |
| word2vec  | LibLinear LR | preTrain dblp v5  | 44.61   | 45.2   | 44.05  | 50.79        | 55.11   | 47.37   | 73.32 | 0.00  | 56.87  | 47.4       | 57.52         | 59.82   |
| word2vec  | LibLinear LR | ACM v9 + bdlp v10 | 47.24   | 48.04  | 46.46  | 49.45        | 52.69   | 46.83   | 72.31 | 0.00  | 58.50  | 48.11      | 56.60         | 53.67   |
| word2vec  | LibLinear LR | bdlp v5           | 46.27   | 47.76  | 44.87  | 50.08        | 54.32   | 46.73   | 72.42 | 0.00  | 53.95  | 48.40      | 56.84         | 58.79   |
| word2vec  | LibLinear LR | bdlp v10          | 47.28   | 47.28  | 47.27  | 50.28        | 53.53   | 47.62   | 73.46 | 0.00  | 59.01  | 49.54      | 57.34         | 54.90   |
| fastText  | LibLinear LR | bdlp v5           | 49.12   | 63.54  | 40.03  | 49.3         | 61.64   | 41.2    | 72.08 | 0.00  | 47.31  | 46.27      | 59.88         | 39.12   |
| fastText  | LibLinear LR | acm v9 + bdlp v10 | 50.21   | 64.11  | 41.27  | 50.73        | 62.45   | 42.79   | 72.35 | 0.00  | 48.74  | 48.97      | 60.93         | 45.76   |
| fastText  | LibLinear LR | bdlp v10          | 50.75   | 63.85  | 42.12  | 49.72        | 61.29   | 41.93   | 72.33 | 0.00  | 49.84  | 46.00      | 60.80         | 41.14   |
| bert      | LibLinear LR | acm v9 + bdlp v10 | 26.02   | 26.02  | 26.02  | 32.82        | 37.23   | 29.74   | 67.18 | 0.00  | 0.00   | 41.39      | 55.09         | 6.92    |

> We found that using fasText embedding on bdlp v10 has the best performance, so we do the further experience on other models

```sh
cd LightRel
python3 loadFeatureAndTrain.py
```

scikit-learn v0.20.3

| Embedding | Model        | Data     | Test F1 | Test P | Test R | Train F1 (%) | Train P | Train R | USAGE | TOPIC | RESULT | PART_WHOLE | MODEL-FEATURE | COMPARE |
| --------- | ------------ | -------- | ------- | ------ | ------ | ------------ | ------- | ------- | ----- | ----- | ------ | ---------- | ------------- | ------- |
| fastText  | LinearSVC    | bdlp v10 | 51.47   | 51.43  | 51.45  | 50.24        | 53.37   | 47.77   | 70.99 | 0.00  | 55.84  | 47.09      | 58.10         | 61.47   |
| fastText  | Sklearn LR   | bdlp v10 | 56.32   | 49.64  | 52.77  | 53.09        | 58.03   | 49.06   | 74.70 | 0.00  | 60.06  | 50.35      | 59.79         | 62.96   |
| fastText  | DecisionTree | bdlp v10 | 35.37   | 38.03  | 36.65  | 39.26        | 38.29   | 40.54   | 61.21 | 6.67  | 34.17  | 38.69      | 48.24         | 38.38   |

#### Add more feature

First, because there are only 18 training sample for TOPIC in training set. We currently get 0 on F1-score.

We list the relation of TOPIC in training set as the following.

```py
from util import loadRelation
from constant import rela2id
relations = loadRelation('data/1.1.relations.txt')
topics_entities = [k for k, v in a.items() if v == rela2id['TOPIC']]

# [('P01-1009.1', 'P01-1009.3'),
# ('N03-1026.14', 'N03-1026.15'),
# ('N04-1024.18', 'N04-1024.19'),
# ('H01-1055.7', 'H01-1055.9'),
# ('E06-1004.1', 'E06-1004.2'),
# ('C80-1073.4', 'C80-1073.5'),
# ('A92-1023.4', 'A92-1023.5'),
# ('A92-1023.7', 'A92-1023.8'),
# ('P06-1053.1', 'P06-1053.2'),
# ('N06-1007.7', 'N06-1007.8'),
# ('E89-1016.1', 'E89-1016.3'),
# ('E93-1013.1', 'E93-1013.2'),
# ('E95-1036.11', 'E95-1036.12'),
# ('H93-1076.3', 'H93-1076.4'),
# ('A97-1028.11', 'A97-1028.12'),
# ('P99-1058.11', 'P99-1058.12'),
# ('X96-1041.6', 'X96-1041.7'),
# ('J87-3001.16', 'J87-3001.17')]
```

1. ('P01-1009.1', 'P01-1009.3')
   * Title: Alternative Phrases and Natural Language Information Retrieval
   * `formal analysis` for a large class of words called `alternative markers`
2. ('N03-1026.14', 'N03-1026.15')
   * Title: Statistical Sentence Condensation using Ambiguity Packing and Stochastic Disambiguation Methods for Lexical-Functional Grammar
   * An `experimental evaluation` of `summarization`
3. ('N04-1024.18', 'N04-1024.19')
   * Title: Evaluating Multiple Aspects of Coherence in Student Essays
   * `Intra-sentential quality` is evaluated with `rule-based heuristics`
4. ('H01-1055.7', 'H01-1055.9')
   * Title: Natural Language Generation in Dialog Systems
   * `system response` to users has been extensively studied by the `natural language generation community`

## Using Deep Learning

> TODO

### Learning from the extracted feature

### Learning from the beginning

---

## Appendix: The performance of random

```sh
python3 random_test.py
```

```txt
<<< RELATION EXTRACTION EVALUATION >>>

Precision = 355/355 = 100.00%
Recall = 355/355 = 100.00%
F1 = 100.00%

<<< The official score for the extraction scenario is F1 = 100.00% >>>


<<< RELATION CLASSIFICATION EVALUATION >>>:

Number of instances in submission: 355
Number of instances in submission missing from gold standard: 0
Number of instances in gold standard:  355
Number of instances in gold standard missing from submission:    0

Coverage (predictions for a correctly extracted instance with correct directionality) = 355/355 = 100.00%

Results for the individual relations:
                  COMPARE :    P =    7/  71 =   9.86%     R =    7/  21 =  33.33%     F1 =  15.22%
            MODEL-FEATURE :    P =   13/  57 =  22.81%     R =   13/  66 =  19.70%     F1 =  21.14%
               PART_WHOLE :    P =    5/  54 =   9.26%     R =    5/  70 =   7.14%     F1 =   8.06%
                   RESULT :    P =    6/  60 =  10.00%     R =    6/  20 =  30.00%     F1 =  15.00%
                    TOPIC :    P =    0/  57 =   0.00%     R =    0/   3 =   0.00%     F1 =   0.00%
                    USAGE :    P =   26/  56 =  46.43%     R =   26/ 175 =  14.86%     F1 =  22.51%

Micro-averaged result :
P =   57/ 355 =  16.06%     R =   57/ 355 =  16.06%     F1 =  16.06%

Macro-averaged result :
P =  16.39%     R =  17.51%     F1 =  16.93%



<<< The official score for the classification scenario is macro-averaged F1 = 16.93% >>>
```

## Trouble Shooting

### Execute shell command and get stdout

```py
import subprocess
```

- [Python: How to get stdout after running os.system?](https://stackoverflow.com/questions/18739239/python-how-to-get-stdout-after-running-os-system/45364515)

### XML

- [How do I parse XML in Python?](https://stackoverflow.com/questions/1912434/how-do-i-parse-xml-in-python)
- [How to extract the text between some anchor tags?](https://stackoverflow.com/questions/13247479/how-to-extract-the-text-between-some-anchor-tags)
- [Remove a tag using BeautifulSoup but keep its contents](https://stackoverflow.com/questions/1765848/remove-a-tag-using-beautifulsoup-but-keep-its-contents)

### JSON

- [Stackoverflow - Read the json data in shell script](https://stackoverflow.com/questions/20488315/read-the-json-data-in-shell-script)

### #include <malloc.h> not found in OSX

- [github RIOT-OS/RIOT issue - malloc.h not found on OS X #2361](https://github.com/RIOT-OS/RIOT/issues/2361#issuecomment-78629199)

### Python Type hint

- [Stackoverflow - How to annotate types of multiple return values?](https://stackoverflow.com/questions/40181344/how-to-annotate-types-of-multiple-return-values)

### Get BERT word embedding

* [google-research/bert issues - How to get the word embedding after pre-training?](https://github.com/google-research/bert/issues/60)
  * with variable name `bert/embeddings/word_embeddings`
* [imgarylai/bert-embedding](https://github.com/imgarylai/bert-embedding)
* [Stackoverflow - How do I find the variable names and values that are saved in a checkpoint?](https://stackoverflow.com/questions/38218174/how-do-i-find-the-variable-names-and-values-that-are-saved-in-a-checkpoint/41917296)
  * [tensorflow/tensorflow - inspect_checkpoint.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/inspect_checkpoint.py)

```py
model_path = '.'
# Get latest checkpoint
latest_ckpt = tf.train.latest_checkpoint(model_path)
```

```py
# just print
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

print_tensors_in_checkpoint_file(latest_ckpt, all_tensors=False, tensor_name='bert/embeddings/word_embeddings')
```

```py
# get the embedding tensor
tensor_name = 'bert/embeddings/word_embeddings'
file_name = latest_ckpt
from tensorflow.python import pywrap_tensorflow
reader = pywrap_tensorflow.NewCheckpointReader(file_name)
embedding = reader.get_tensor(tensor_name) # np.array (default dimension 768)

# The length should be the same as vocab.txt
len(embedding) # 30522 (default)
```

```py
# combine into embedding file (vocabulary: tensor)
with open('vocab.txt', 'r') as vocab:
    words = vocab.readlines()

test = [' '.join([str(combined) for combined in [word.strip(), *embedding[i]]]) for i, word in enumerate(words)]
```

### Random stuff

- [How do I remove a substring from the end of a string in Python?](https://stackoverflow.com/questions/1038824/how-do-i-remove-a-substring-from-the-end-of-a-string-in-python)

```py
import module
from imp import reload

reload(module) # module updated
```

- [Difference between **str** and **repr**?](https://stackoverflow.com/questions/1436703/difference-between-str-and-repr)

- [Finding All The Keys With the Same Value in a Python Dictionary](https://stackoverflow.com/questions/42438808/finding-all-the-keys-with-the-same-value-in-a-python-dictionary)

## TODO

- [ ] feature dimension problem in 5-fold evaluation on version 1.2 dataset

## Links

### TextCNN

* [brightmart/text_classification](https://github.com/brightmart/text_classification)
* [DongjunLee/text-cnn-tensorflow](https://github.com/DongjunLee/text-cnn-tensorflow)

### Data

- Competition
  - [SemEval-2018 task 7 Semantic Relation Extraction and Classification in Scientific Papers](https://competitions.codalab.org/competitions/17422)
- Data
  - [SemEval 2018 Task 7](https://lipn.univ-paris13.fr/~gabor/semeval2018task7/)
  - [gkata/SemEval2018Task7](https://github.com/gkata/SemEval2018Task7/tree/testing)

### Paper

- [SemEval-2018 Task 7: Semantic Relation Extraction and Classification in Scientific Papers](https://aclweb.org/anthology/S18-1111)
- [LightRel](https://arxiv.org/pdf/1804.08426.pdf)
  - [github](https://github.com/trenslow/LightRel)
