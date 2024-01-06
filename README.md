# scientific_paper_classification

<a href="https://arxiv.org/">ArXiv</a> is a public repository of scientific papers where researchers and students from anywhere in the world can find the latest results in many disciplines ranging from Natural Sciences to Mathematics and Computer Science. The papers are categorized by discipline and (possibly multiple) subdiscipline. The platform also displays each paper's abstract without the need to download the file.

The categorization of a new paper is important for authors to make sure their research will reach the intended audience and for the readers to find the most relevant works in their interest area. Presently the categorization process is the authors' sole responsibility. This fact raises some questions:
<ol>
    <li>Is the category chosen for a research paper the most appropriate?</li>
    <li>Is it possible to make a reasonable prediction of a paper's category given only a summary or it's abstract?</li>
</ol>

In this project we display how a Large Language Model can be leveraged to help classify a scientific paper. We will fine tune a pretrained encoder model to predict paper's category(ies).

## Dataset

The dataset is <a href="https://www.kaggle.com/datasets/Cornell-University/arxiv">ArXiv Dataset (Kaggle)</a> retrieved at December 20th, 2023. There are 2385180 documents in this dataset. Their structure is explained in the Data Card at the source.
