# Empirical Study on the Scaling Laws of Human Genomic Language Models
In recent years, the application of natural language processing techniques to genomic analysis has garnered significant attention, particularly through the development of genomic language models. This study investigates the scaling laws of human genomic language models, specifically focusing on DNABERT-2 as a foundational model. We explore the relationships between model performance, parameter size, data volume, and computational resources, revealing that human genomic language models exhibit distinct scaling laws. Our findings indicate that for human genomic language models, increasing the parameter size is more crucial than expanding the data volume, in contrast to natural language models. Furthermore, we analyze the variations in scaling laws under different sampling and tokenization strategies for DNA sequences. This research aims to provide a foundational understanding of scaling laws in human genomic language models, offering valuable insights for future genomic studies and model development.

## 1. Datasets

### 1.1 The human reference genome dataset
The human reference genome dataset we use is based on the genome assembly GRCh38, including autosomal and sex chromosome sequences, with a total length of approximately 3.2 billion nucleotides. This dataset provides a relatively comprehensive and accurate human genome reference and is widely used in various genome studies.

Download [the human reference genome dataset](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.26/) and place it in the ```data``` folder.

### 1.2 The human pan-genome dataset
The human pan-genome dataset we used is based on the 1000 Genomes Project, an international research initiative aimed at creating the most detailed catalog of human genetic variation. The data from the 1000 Genomes Project are widely utilized in research and are freely accessible to the public. We downloaded the variant calling format (VCF) files from the project, which document genetic variants with a frequency of 1% or greater in the population. These variant data statistics are derived from 3,202 high-coverage human genomes, encompassing a total of 20.5 trillion nucleotides. This dataset contains rich information on genetic variants, and we believe that such diverse data can more comprehensively represent human genetic variation.

Download  [the human pan-genome dataset](http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_phased/) and place it in the ```data``` folder.

### 1.3 Data preparation

We employed three different sampling strategies on the downloaded corpus, resulting in three corresponding pre-training datasets. The three sampling strategies include random sampling from the human reference genome, random sampling from the human pan-genome, and random sampling from variant sequences of the human pan-genome.

Please see ```gene_library_construction/main.py```. The output results will be placed in the following three folders:```gene_library_construction/random_sampling/```,```gene_library_construction/reference_sampling/```,```gene_library_construction/variant_sampling/```.

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