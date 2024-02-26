# multimodal_lucid
Code for multimodal model training using multiple data fusion techniques for LUCID (Low-dose Understanding, Cellular Insights, and Molecular Discoveries).

Even though LUCID specifically focuses on low does settings, since we do not have access to clean LDR-induced cancer specific datasets right now, we are using comparable datasets from the public TCGA
data resource. 


Task:

The objective is to develop a multimodal model to predict event times for cancer cases using matched WSI and molecular data from the TCGA database.
Currently, we are using data from the lung adenocarcinoma samples from the TCGA-LUAD database 

Data used:

1. Histology data [Whole Slide Images (WSI)] <br />


2. Tabular molecular data: gene expression (bulk RNA-seq) <br />

[//]: # (These data have been collected from https://drive.google.com/drive/folders/14TwYYsBeAnJ8ljkvU5YbIHHvFPltUVDr)


Method:
Embeddings are generated from the WSI data and the tabular molecular features using a CNN and a MLP, respectively, that are fused to be used as input for a downstream MLP that has its final node predicting the log-risk score (log of the hazard ratio) for the Cox log partial likelihood function representing the loss function. In the joint fusion approach, all the models are trained simultaneously using the loss function.
We are also exploring other methods for embedding generation, including attention based encoder models.
