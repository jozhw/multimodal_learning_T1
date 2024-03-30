# Multimodal data fusion strategies for LUCID thrust 1
Code for multimodal model training using multiple data fusion techniques for LUCID (Low-dose Understanding, Cellular Insights, and Molecular Discoveries). Even though LUCID specifically focuses on low dose settings, in the absence of clean LDR-induced cancer specific datasets (that should be made available shortly), we are using comparable datasets from the public TCGA database. 

## Table of content

- [Task](#task)
- [Data used](#data-used)
- [Training methodlogy](#training-methodology)
- [Running the code](#running-the-code)
- [Code structure](#code-structure)

## Task

The objective is to develop a deep learning framework to predict cancer-related times to events using matched multimodal samples. Currently, we are using whole slide image (WSI) and tabular gene expression (bulk RNASeq) data from the lung adenocarcinoma samples from the TCGA database (TCGA-LUAD). The code is being developed in a modular framework so that it can be easily extended to handle more input data modalities when the need arises. 

## Data used:

1. Histology data [Whole Slide Images (WSI)] <br />


2. Tabular gene expression (bulk RNA-seq) data <br />

[//]: # (These data have been collected from https://drive.google.com/drive/folders/14TwYYsBeAnJ8ljkvU5YbIHHvFPltUVDr)


## Training methodology
Embeddings are generated from the WSI data and the tabular molecular features using a CNN and a MLP, respectively, that are fused to be used as input for a downstream MLP that has its final node predicting the log-risk score (log of the hazard ratio) for the Cox log partial likelihood function representing the loss function. In the joint fusion approach, all the models are trained simultaneously using the loss function.
We are also exploring other methods for embedding generation, including attention based encoder models.

1. Embedding generation from WSI data (at the tile level)
2. Embedding generation from the gene expression data
3. Fusing embeddings from all modalities for downstream MLP training for cancer-specific time to event prediction
4. Loss function and model training

## Running the code

``` sh
python trainer.py --input_path --input_wsi_path --batch_size --lr --lr_decay_iters --num_epochs --gpu_ids --input_size_wsi --embedding_dim_wsi --embedding_dim_omic --input_modes --fusion_type --profile --use_mixed_precision --use_gradient_accumulation

```


| Input Arguments      | Explanation                                             | Type and Default Value                                                               |
|----------------------|---------------------------------------------------------|--------------------------------------------------------------------------------------|
| --input_path         | Path to input data files                               | type=str, default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/' |
| --input_wsi_path     | Path to input WSI tiles                                | type=str, default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/TCGA_WSI/batch_corrected/processed_svs/tiles/256px_9.9x/combined_tiles/' |
| --batch_size         | Batch size for training                                | type=int, default=4                                                                   |
| --lr                 | Learning rate                                          | type=float, default=0.001                                                             |
| --lr_decay_iters     | Learning rate decay steps                              | type=int, default=100                                                                 |
| --num_epochs         | Number of training epochs                              | type=int, default=2                                                                   |
| --gpu_ids            | gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU            | type=str, default='0'                                                                 |
| --input_size_wsi     | input_size for path images                             | type=int, default=256                                                                 |
| --embedding_dim_wsi  | embedding dimension for WSI                            | type=int, default=128                                                                 |
| --embedding_dim_omic | embedding dimension for omic                           | type=int, default=256                                                                 |
| --input_modes        | wsi, omic, wsi_omic                                    | type=str, default="wsi"                                                               |
| --fusion_type        | early, late, joint, unimodal                           | type=str, default="unimodal"                                                          |
| --profile            | whether to profile or not                              | type=str, default=False                                                               |
| --use_mixed_precision| whether to use mixed precision calculations            | type=str, default=False                                                               |
| --use_gradient_accumulation | whether to use gradient accumulation               | type=str, default=False                                                               |


## Code structure
