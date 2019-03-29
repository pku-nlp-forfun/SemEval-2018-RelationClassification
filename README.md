# SemEval-2018-RelationClassification

A PKU course project based on the "SemEval-2018 task 7 Semantic Relation Extraction and Classification in Scientific Papers" dataset.

The Subtask 1

* 1.1 Relation classification on clean data
* 1.2 Relation classification on noisy data

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

* `USAGE`: 483
* `TOPIC`: 18
* `RESULT`: 72
* `PART_WHOLE`: 234
* `MODEL-FEATURE`: 326
* `COMPARE`: 95

### Subtask 1: Relation classification

> For each subtask, training and test data include abstracts of papers from the ACL Anthology Corpus with pre-annotated entities that represent concepts. Two types of tasks are proposed:
>
> 1) identifying pairs of entities that are instances of any of the six semantic relations (extraction task),
>
> 2) classifying instances into one of the six semantic relation types (classification task).

The subtask is decomposed into two scenarios according to the data used: classification on clean data and classification on noisy data. The task is identical for both scenarios: given a relation instance consisting of two entities in context, predict the semantic relation between the entities. A relation instance is identified by the unique ID of the two entities.

For the subtask 1,  instances with directionality are provided in both the training data and the test data and they are not to be modified or completed in the test data; the relation label is provided in the training data and has to be predicted for the test data.

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

* for every distinct class: precision, recall and F1-measure (β=1)
* global evaluation, for the set of classes:
  * macro-average of the F1-measures of every distinct class
  * micro-average of the F1-measures of every distinct class

## Result

### Using Traditional ML Method

#### Procedure

1. Load dataset
   1. Load the *relations* into a hash map of `(entity 1, entity 2) -> relation`
   2. Load the *text* into a Paper class object
      * paper id
      * title
      * abstract (plain text)
      * entity id and text
2. Sentence embedding => Get the training feature with label
3. Training model
   * SVM
   * Logistic Regression
4. Test the result
   * Using the SemEval 2018 task 7 scorer
   * Calculate by ourself (5-fold cross validation)

#### The Idea of sentence embedding

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

### Using Pure Embedding Model

#### Another way of sentence embedding

Calculate three parts of embedding. `entity 1`, `relationship`, and `entity 2`.

> And two of them, doing dot product. Then we'll get 2 numbers. And use them to form the result.

### Using LightRel as Baseline - [Forked Project](https://github.com/pku-nlp-forfun/LightRel)

* LightRel trained the dictionary for SemEval 2010 Task 8 and reused it for SemEval 2010 Task 7

#### Addition library used

* [liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
* [ast](https://docs.python.org/3/library/ast.html) - Abstract Syntax Trees
* [unicodedata](https://docs.python.org/3/library/unicodedata.html)

#### Conclusion

The model used by LightRel is not complicated. The performance is depend more on features (embedding).

### The performance of random

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

* [Python: How to get stdout after running os.system?](https://stackoverflow.com/questions/18739239/python-how-to-get-stdout-after-running-os-system/45364515)

### XML

* [How do I parse XML in Python?](https://stackoverflow.com/questions/1912434/how-do-i-parse-xml-in-python)
* [How to extract the text between some anchor tags?](https://stackoverflow.com/questions/13247479/how-to-extract-the-text-between-some-anchor-tags)
* [Remove a tag using BeautifulSoup but keep its contents](https://stackoverflow.com/questions/1765848/remove-a-tag-using-beautifulsoup-but-keep-its-contents)

### Random stuff

* [How do I remove a substring from the end of a string in Python?](https://stackoverflow.com/questions/1038824/how-do-i-remove-a-substring-from-the-end-of-a-string-in-python)

```py
import module
from imp import reload

reload(module) # module updated
```

* [Difference between __str__ and __repr__?](https://stackoverflow.com/questions/1436703/difference-between-str-and-repr)

## Links

### Data

* Competition
  * [SemEval-2018 task 7 Semantic Relation Extraction and Classification in Scientific Papers](https://competitions.codalab.org/competitions/17422)
* Data
  * [SemEval 2018 Task 7](https://lipn.univ-paris13.fr/~gabor/semeval2018task7/)
  * [gkata/SemEval2018Task7](https://github.com/gkata/SemEval2018Task7/tree/testing)

### Paper

* [SemEval-2018 Task 7: Semantic Relation Extraction and Classification in Scientific Papers](https://aclweb.org/anthology/S18-1111)
* [LightRel](https://arxiv.org/pdf/1804.08426.pdf)
  * [github](https://github.com/trenslow/LightRel)
