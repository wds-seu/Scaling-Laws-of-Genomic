# Empirical Study on the Scaling Laws of Human Genomic Language Model
In recent years, the application of natural language processing techniques to genomic analysis has garnered significant attention, particularly through the development of genomic language models. However, the optimal allocation of data, model parameters, and computational resources for enhancing the understanding of gene sequences remains unclear. This study investigates the scaling laws of the state-of-the-art human genomic language model, exploring the relationships between model performance, parameter size, data volume, and computational resources. Our findings reveal that human genomic language models follow distinct scaling laws, with increases in parameter size being more critical than data volume expansion, in contrast to natural language models. Additionally, we analyze how different sampling and tokenization strategies for DNA sequences influence the scaling laws. This research seeks to provide a foundational and nuanced understanding of scaling laws in human genomic language models, offering valuable insights for future genomic research and model development.

## 1. Datasets

### 1.1 The human reference genome dataset
The human reference genome dataset we use is based on the genome assembly GRCh38, including autosomal and sex chromosome sequences, with a total length of approximately 3.2 billion nucleotides. This dataset provides a relatively comprehensive and accurate human genome reference and is widely used in various genome studies.

Download [the human reference genome dataset](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.26/) and place it in the ```data``` folder.

### 1.2 The human pan-genome dataset
The human pan-genome dataset used in this study is derived from the 1000 Genomes Project, an initiative aimed at cataloging human genetic variation. The publicly accessible data from this project are widely used in research. We downloaded the variant calling format (VCF) files documenting genetic variants across 3,202 high-coverage human genomes. This dataset provides rich information on genetic variants, enabling a more comprehensive representation of human genetic variation.

Download  [the human pan-genome dataset](http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_phased/) and place it in the ```data``` folder.

### 1.3 Data preparation

We constructed pre-training datasets using the human reference genome and human pan-genome datasets to investigate the scaling laws of human genomic language models. Three sampling strategies were applied to the downloaded corpus. Each sampling strategy ultimately results in one billion sampled DNA sequences.

Please see ```gene_library_construction/main.py```. The output results will be placed in the following three folders:```gene_library_construction/pan-genome_sampling/```,```gene_library_construction/reference_sampling/```,```gene_library_construction/variation_sampling/```.

## 2. Model Pre-training

### 2.1 Environment setup

Please see ```DNABERT_2/requirements.txt```.

### 2.2 Modify configuration files and hyperparameters
The configuration file is ```DNABERT_2/config.json```, which can be used to modify the hidden layer dimensions, number of layers, and other configurations of the DNABERT-2 model. Hyperparameters during training are set in ```DNABERT_2/parse.py```.

### 2.3 Data processing
Please see ```DNABERT_2/dataset.py```.

### 2.4 Pretraining
Please see ```DNABERT_2/pretrain.py```.


## 3. Plot Results
Please see ```DNABERT_2/loss_plot.py```. The final result is in the ```figs``` folder