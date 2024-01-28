# scientific_paper_classification

<a href="https://arxiv.org/">ArXiv</a> is a public repository of scientific papers where researchers and students from anywhere in the world can find the latest results in many disciplines ranging from Natural Sciences to Mathematics and Computer Science. The papers are categorized by discipline and (possibly multiple) subdiscipline. The platform also displays each paper's abstract without the need to download the file.

The categorization of a new paper is important for authors to make sure their research will reach the intended audience and for the readers to find the most relevant works in their interest area. Presently the categorization process is the authors' sole responsibility. This fact raises some questions:
<ol>
    <li>Is the category chosen for a research paper the most appropriate?</li>
    <li>Is it possible to make a reasonable prediction of a paper's category given only a summary or it's abstract?</li>
</ol>

In this project we display how a Large Language Model can be leveraged to help classify a scientific paper. We will employ the transfer learning technique using a pretrained transformer model to predict paper's categories.


## Dataset

The dataset is <a href="https://www.kaggle.com/datasets/Cornell-University/arxiv">ArXiv Dataset (Kaggle)</a> retrieved at December 20th, 2023. There are 2385180 documents in this dataset. Their structure is explained in the Data Card at the source.

To simplify the tuning and evaluation, we have only selected papers categorized as either <em>Computer Science - Artificial Intelligence (cs.AI)</em> or <em>Computer Science - Machine Learning (cs.LG)</em>. Some papers are categorized with both labels. Besides, we selected only papers uploaded since the beginning of 2022.

The resulting dataset sizes (after train-test-split and tokenization) can be seen in the code blocks below.

#### Train - Validation - Test Split

<pre><code>
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'categories', 'abstract', 'cs.AI', 'cs.LG'],
        num_rows: 49777
    })
    validation: Dataset({
        features: ['id', 'title', 'categories', 'abstract', 'cs.AI', 'cs.LG'],
        num_rows: 12445
    })
    test: Dataset({
        features: ['id', 'title', 'categories', 'abstract', 'cs.AI', 'cs.LG'],
        num_rows: 15556
    })
})
</code></pre>

#### Tokenized

<pre><code>
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'categories', 'abstract', 'cs.AI', 'cs.LG', 'text', 'input_ids', 'attention_mask'],
        num_rows: 124846
    })
    validation: Dataset({
        features: ['id', 'title', 'categories', 'abstract', 'cs.AI', 'cs.LG', 'text', 'input_ids', 'attention_mask'],
        num_rows: 31186
    })
    test: Dataset({
        features: ['id', 'title', 'categories', 'abstract', 'cs.AI', 'cs.LG', 'text', 'input_ids', 'attention_mask'],
        num_rows: 39200
    })
})
</code></pre>

From each paper the features are the concatenation of the texts in the <code>title</code> and <code>abstract</code> columns. The joined texts were tokenized using the model described below.


## Model

The model used was <a href="https://huggingface.co/distilbert-base-uncased">distilbert-base-uncased</a>, a distilled version of BERT. The transfer learning used pytorch to update the model's head.


## Pipeline

Here I briefly describe the steps of the transfer learning and final classification:

<ol>
    <li>script <a href="https://github.com/MarcosBrum/scientific_paper_classification/blob/main/source/make_train_eval_test_dataset.py">make_train_eval_test_dataset</a>: open raw dataset, select categories and separate <code>train-evaluation-test</code> datasets;</li>
    <li>script <a href="https://github.com/MarcosBrum/scientific_paper_classification/blob/main/source/prepare_data.py">prepare_data</a>: tokenize texts in <code>train-evaluation-test</code> datasets;</li>
    <li>script <a href="https://github.com/MarcosBrum/scientific_paper_classification/blob/main/source/fine_tune_pretrained_model.py">fine_tune_pretrained_model</a>: perform transfer learning using prepared data in <code>train</code> dataset and evaluate model using <code>evaluation</code> dataset;</li>
    <li>script <a href="https://github.com/MarcosBrum/scientific_paper_classification/blob/main/source/test_tuned_model.py">test_tuned_model</a>: test model's predictions using <code>test</code> dataset.</li>
</ol>

The notebook <a href="https://github.com/MarcosBrum/scientific_paper_classification/blob/main/notebooks/view_data.ipynb">view_data</a> provides a visualization of the original dataset and distribution of category quantities. The notebook <a href="https://github.com/MarcosBrum/scientific_paper_classification/blob/main/notebooks/multilabel_confusion_matrix.ipynb">multilabel_confusion_matrix</a> contains the confusion matrices displaying the classification results.


## Results

<p align="center">
    <img src="https://github.com/MarcosBrum/scientific_paper_classification/blob/main/confusion_matrices/cs_ai_confusion_matrix.png" height="270" width="360" hspace="10" title="Confusion matrix for Artificial Intelligence category"/>
    <img src="https://github.com/MarcosBrum/scientific_paper_classification/blob/main/confusion_matrices/cs_lg_confusion_matrix.png" height="270" width="360" hspace="10" title="Confusion matrix for Machine Learning category"/>
</p>

<pre><code>
========== Classification for label Artificial Intelligence ==========
precision = 0.43
recall    = 0.55
f1_score  = 0.48

========== Classification for label Machine Learning =================
precision = 0.69
recall    = 0.69
f1_score  = 0.69
</code></pre>

As the results display, the classification of <code>Machine Learning</code> papers achieved a better performance but the results for both classes lie below expectations.

In order to improve the results, here is what I would try:

<ul>
    <li>Get more data: make transfer learning using data from a longer period.</li>
    <li>Select data: search for outliers (and remove from training/evaluation dataset); filter highly similar data.</li>
    <li>Fine tune hyperparameters.</li>
    <li>Let transfer learning run for more epochs.</li>
</ul>
