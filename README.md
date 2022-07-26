# MSc Thesis Project Transparency Lab 
This repository contains the code base of the Master Thesis project for the Master Project AI at the Vrije Universiteit (VU), Amsterdam.


In this Master Thesis project, we propose a framework for the preparation of textual documents for Questionnaire Generation. 

| Model                 | Description |
| ----------------------| ------------- |
| Data & Score Model    | The texutal documents are assessed on a number of criteria. The four criteria that add to the Story-score are (1) Relevance, (2)  Additions, (3) Contrasts, and (4) Comparisons. The second group of criteria say something about the structure of a paper, (1) Frequency of numbers, (2) Frequency of question marks, (3) the occurrence of SMART objectives, and (4) Overall document quality. Based on the assessment of these criteria on the documents, a set of qualitative documents is obtained.|
| Topic Model           | The qualitative documents are clustered using LDA to obtain document corpora  |
| Language Model        | One of the corpora obtained with the Topic model is used as corpus in TF-IDF to extract keywords from an input document                 | 

## Prerequisites 
1. An x86 machine running Windows 10 or a Unix-based OS
2. Python 3.7 or higher
