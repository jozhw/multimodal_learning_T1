# multimodal_lucid
Code for multimodal model training using multiple data fusion techniques for LUCID (Low-dose Understanding, Cellular Insights, and Molecular Discoveries).

Even though LUCID specifically focuses on low does settings, since we do not have access to clean LDR-induced cancer specific datasets right now, we are using comparative datasets from the public TCGA
data resource. 


Task:

The objective is to develop a multimodal model to predict cancer patient survival outcome (i.e., time to death) using matched WSI and molecular data from the TCGA database.

Currently, we are using data corresponding to Low grade glioma (LGG) (WHO grades II and III), Glioblastoma (GBM) (WHO grade IV), and Kidney clear cell renal cell carcinoma (KIRC). TCGA contains paired gene expression (bulk RNA-seq) and diagnostic whole slide images (WSI) with ground-truth survival outcome and histologic grade labels.

WHO currently classifies diffuse gliomas based on morphological and molecular characteristics: glial cell type (astrocytoma, oligodendroglioma), IDH1 gene mutation status
and 1p/19q chromosome codeletion status. So, in addition to the gene expression data, these molecular features will also be used as inputs for model training for the LGG and GBM cases.


Data used:

1. Histology data [Whole Slide Images (WSI)] <br />


2. Tabular molecular data: gene expression (bulk RNA-seq), copy number variation and mutation data <br />

These data have been collected from https://drive.google.com/drive/folders/14TwYYsBeAnJ8ljkvU5YbIHHvFPltUVDr


Method:
Embeddings are generated from the WSI data and the tabular molecular features using a CNN and a MLP, respectively, that are fused to be used as input for a downstream MLP that has it's final node predicting the log-risk score (log of the hazard ratio) for the Cox log partial likelihood function representing the loss function. In the joint fusion approach, all the models are trained simultaneously using the loss function.
