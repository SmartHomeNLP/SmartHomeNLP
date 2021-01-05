# Smart Home NLP

This project investigates language usage surrounding Smart Home 
technologies on Reddit forums "smarthome" and "homeautomation". 
The discussion around privacy, security and trust regarding 
these technologies receives especial focus. The project uses
topic modeling (LDA) to accomplish this. 

## Pipeline: (work in progress)

All files are located within the SmartHome folder. 

### Download data 
Navigate to pushshift for reddit: https://files.pushshift.io/reddit/.
Download submissions from: https://files.pushshift.io/reddit/submissions/ 
and comments from: https://files.pushshift.io/reddit/comments/. 

### Decompress data
For now the pipeline is set up to handle decompression of 
.zst files (the most recent format used). Navigate to the "decompress_files" folder 
in the project. The decompressed files are written as .csv files into the "data/raw"
folder. 

### Preprocessing 
All preprocessing is done in the "preprocessing" folder. 
The preprocessing pipeline can be run from the "preprocess_pipeline.py"
document. All functions used in "preprocessing_pipeline" are from the "functions.py"
document. The "preprocessing_pipeline" is best run from command line. 

1. The first step of preprocessing is to merge the files from submissions and 
comments. The functions assume that the files have been generated following the 
pipeline (i.e. the function is sensitive to file-names and file-placement). 

All subsequent preprocessing functions in "preprocessing_pipeline" accept 
the argument "done" which can be set to "True" to skip specific steps. 

2. The next step of preprocessing is to exclude bots from the analysis,
with the function "bot_clean". There
is both a manual and an automatic screening. Still a work in progress. 
If only automatic cleaning is desired, the argument 
"manual = False" can be supplied to the function. 

3. The next step of preprocessing is done with the "concat_data" function. 
This function returns a dataframe which retains the necessary structure
for further analysis (e.g. differentiates between comments and submissions). 
NB: difficult to explain concisely (come back, perhaps figure..?). 

4. The next step of preprocessing is done with the "small_cleaning" function
which un-escapes html and removes hyphenation and treats these words as one word. 
(explain better). 

5. The next step of preprocessing is "clean_comment" which is where all the 
actual preprocessing happens. Removes nltk stopwords (and additional ones), 
uses nltk language detection (ensuring 70% english), removes formatting and
punctuation and lemmatizes each word using Spacy. 

6. The last step of preprocessing is done with the "gen_subsets" function. It returns three distinct dataframes; (1) H1_thread which is ... (2)
H1_tree which is ... and (3) H2_submissions which is .... It takes an argument
called "get" which should be a list of the subsets that the user wishes to 
generate. ["thread", "tree", "submission"] or just some of these? 

7. The last function writes dataframes for the three subsets to the folder "data/clean". 

### NLP modeling 
all LDA modeling is done from within the "NLP_modeling" folder. 
The models can be generated from the "LDA_modeling.py" file and rely on functions
from the "model_fun.py" document. 

1. modeling preparation: from "dct_corpus_gen.py" generate the dictionaries
and corpora for the three subsets of data; "submissions", "thread" and "tree". 

2. get back to this (redesign this to be easier to **tweak**). 

3. The last function writes a dictionary into "data/models" 
containing the Gensim models that have been generated. The names of 
the models (keys) reflect the hyperparamters, following the pattern:
a{value}_b{value}_k{value}, where $a = \alpha$, $b = \beta$ and 
$k =$ number of topics. 

### NLP evaluation
All evaluation of LDA models happens inside the folder "NLP_evaluation". 
The files "gen_evaluation.py" and "plot_eval_metrics.py" rely on functions from 
"evaluation_functions.py". 

1. Evaluation metrics "c_v coherence" from Gensim as well "[Arun et al., (2010)](https://link.springer.com/chapter/10.1007/978-3-642-13657-3_43)"
and "[Cao et al., (2009)](https://www.sciencedirect.com/science/article/pii/S092523120800372X?casa_token=pgLkNhzwqGoAAAAA:G51AiUtCIWm8Xy0WvEtws_ckwCS0Gi8m-66YHJ5kvAxTVYQsNBz97Rdsd85A-Ot_5kC7mD1Hwtg)" are generated from "gen_evaluation.py". This returns a dataframe which is ready 
for visualization. (redesign this to be easier to **tweak**)

2. From "plot_eval_metrics" the generated evaluation metrics can be visualized
to find a good model (**tweak**). 

### NLP inspection
All inspection of the topics of the selected model(s) from the 
evaluation step happens in the "NLP_inspection" folder. The documents rely on the 
"inspection_fun.py" document. You now have the LDA models and quantitative metrics
of their quality. The rest of the documents are specifically tailored towards
the plots and the aims of the research questions that we pursue. 

### NLP visualization
Not documented yet. 

## Authors

* **Victor Møller** - *Main Author* (https://github.com/victor-m-p)
* **Mikkel Werling** - *Main Author* (https://github.com/Gotticketsto360tour)

## Acknowledgments

* **Morena Rivato** - *Initial Work* (find Github)