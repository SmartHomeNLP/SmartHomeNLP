# Smart Home NLP

This project investigates language usage surrounding Smart Home 
technologies on Reddit forums "smarthome" and "homeautomation". 
The discussion around privacy, security and trust regarding 
these technologies receives especial focus. The project uses
topic modeling (LDA) to accomplish this. 

## Pipeline: 

All files are located within the SmartHome folder. 

1. Download data: Navigate to pushshift for reddit: https://files.pushshift.io/reddit/.
Download submissions from: https://files.pushshift.io/reddit/submissions/ 
and comments from: https://files.pushshift.io/reddit/comments/. 

2. Decompress data: For now the pipeline is set up to handle decompression of 
.zst files (the most recent format used). Navigate to the "decompress_files" folder 
in the project. The decompressed files are written as .csv files into the "data/raw"
folder. 

3. Preprocessing: All preprocessing is done in the "preprocessing" folder. 
The preprocessing pipeline can be run from the "preprocess_pipeline.py"
document. All functions used in "preprocessing_pipeline" are from the "functions.py"
document. The "preprocessing_pipeline" is best run from command line. 

3.1. The first step of preprocessing is to merge the files from submissions and 
comments. The functions assume that the files have been generated following the 
pipeline (i.e. the function is sensitive to file-names and file-placement). 

All subsequent preprocessing functions in "preprocessing_pipeline" accept 
the argument "done" which can be set to "True" to skip specific steps. 

3.2. The next step of preprocessing is to exclude bots from the analysis,
with the function "bot_clean". There
is both a manual and an automatic screening. Still a work in progress. 
If only automatic cleaning is desired, the argument 
"manual = False" can be supplied to the function. 

3.3. The next step of preprocessing is done with the "concat_data" function. 
This function returns a dataframe which retains the necessary structure
for further analysis (e.g. differentiates between comments and submissions). 
NB: difficult to explain concisely (come back, perhaps figure..?). 

3.4. The next step of preprocessing is done with the "small_cleaning" function
which un-escapes html and removes hyphenation and treats these words as one word. 
(explain better). 

3.5. The next step of preprocessing is "clean_comment" which is where all the 
actual preprocessing happens. Removes nltk stopwords (and additional ones), 
uses nltk language detection (ensuring 70% english), removes formatting and
punctuation and lemmatizes each word using Spacy. 

3.6. The last step of preprocessing is done with the "gen_subsets" function. It 
returns three distinct dataframes; (1) H1_thread which is ... (2)
H1_tree which is ... and (3) H2_submissions which is .... It takes an argument
called "get" which should be a list of the subsets that the user wishes to 
generate. ["thread", "tree", "submission"] or just some of these? 

3.7. The last function writes dataframes for the three subsets to the folder "data/clean". 

4. NLP modeling: all LDA modeling is done from within the "NLP_modeling" folder. 
The models can be generated from the "LDA_modeling.py" file and rely on functions
from the "model_fun.py" document. 

4.1. modeling preparation: from "dct_corpus_gen.py" generate the dictionaries
and corpora for the three subsets of data; "submissions", "thread" and "tree". 

4.2. get back to this (redesign this to be easier to **tweak**). 

4.3 The last function writes a dictionary into "data/models" 
containing the Gensim models that have been generated. The names of 
the models (keys) reflect the hyperparamters, following the pattern:
a{value}_b{value}_k{value}, where $a = \alpha$, $b = \beta$ and 
$k =$ number of topics. 

5. NLP evaluation: All evaluation of LDA models happens inside the folder "NLP_evaluation". 
The 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Victor Møller** - *Main author* (https://github.com/victor-m-p)
* **Mikkel Werling** - *Main author* (https://github.com/Gotticketsto360tour)

## Acknowledgments

* **Morena Rivato** - *Initial work* (find Github)
* Inspiration?