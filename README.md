# multimodal_lucid
Code for multimodal model training for LUCID (Low-dose Understanding, Cellular Insights, and Molecular Discoveries)


Primary cancer types:

1. Low grade glioma (LGG) [WHO grades II and III]
2. Glioblastoma (GBM) [WHO grade IV]
3. Kidney clear cell renal cell carcinoma (KIRC)

Data used:

1. Histology Whole Slide Images (WSI) from TCGA <br />
    -- can use 256 x 256 or 1024 x 1024 patches


2. Genomic data 
    -- bulk RNA-Seq expression from the top 240 differentially expressed genes (240)
       [can add  features like CNV for specific genes and IDH1 gene mutation status] 

Task:

1. Cancer grade classification
2. Survival analysis using Cox PH model
