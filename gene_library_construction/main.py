import time
import pysam
import random
import multiprocessing

MAX_LENGTH_OF_SEQ = 700  # 序列长度
NUMBER_OF_SEQ = 100000000  # 序列总数量
NUMBER_OF_FILE = 100  # 写入文件数目
fasta = pysam.FastaFile('../data/GCF_000001405.26_GRCh38_genomic.fna')
lock_of_fna = multiprocessing.Lock()
locks_of_vcf = [multiprocessing.Lock() for _ in range(23)]
locks_of_write = [multiprocessing.Lock() for _ in range(NUMBER_OF_FILE)]

chromosomes = {  # 染色体
    "chr1": "NC_000001.11", "chr2": "NC_000002.12", "chr3": "NC_000003.12", "chr4": "NC_000004.12",
    "chr5": "NC_000005.10", "chr6": "NC_000006.12", "chr7": "NC_000007.14", "chr8": "NC_000008.11",
    "chr9": "NC_000009.12", "chr10": "NC_000010.11", "chr11": "NC_000011.10", "chr12": "NC_000012.12",
    "chr13": "NC_000013.11", "chr14": "NC_000014.9", "chr15": "NC_000015.10", "chr16": "NC_000016.10",
    "chr17": "NC_000017.11", "chr18": "NC_000018.10", "chr19": "NC_000019.10", "chr20": "NC_000020.11",
    "chr21": "NC_000021.9", "chr22": "NC_000022.11", "chrX": "NC_000023.11",
}

chromosomes_length = {  # 染色体长度
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983, "chr22": 50818468, "chrX": 156040895,
}

vcf = {}
for chrom_i in chromosomes.keys():
    if chrom_i != 'chrX':
        vcf_file = '../data/CCDG_14151_B01_GRM_WGS_2020-08-05_' + chrom_i + '.filtered.shapeit2-duohmm-phased.vcf.gz'
    else:
        vcf_file = '../data/CCDG_14151_B01_GRM_WGS_2020-08-05_chrX.filtered.eagle2-phased.v2.vcf.gz'
    vcf[chrom_i] = pysam.VariantFile(vcf_file, 'rb')


# 查询某一个体的某一个单倍体的某一条染色体上的一段序列[start,end]，以及是否变异
def query_haplotype(sample_name, haplotype, chrom_id, start, end):
    try:
        chrom = list(chromosomes.keys())[chrom_id]
        with lock_of_fna:
            ref_sequence = fasta.fetch(chromosomes[chrom], start - 1, end).upper()  # (L,R]
        if sample_name is None:
            return ref_sequence, True

        variants = []
        with locks_of_vcf[chrom_id]:
            for record in vcf[chrom].fetch(chrom, start - 1, end):
                if sample_name in record.samples:
                    genotype = record.samples[sample_name]['GT']
                    if genotype is not None and genotype[haplotype] == 1:
                        alt = record.alts[0]
                        variants.append((record.pos, record.ref, alt))

        if len(variants) == 0:
            return ref_sequence, True

        # 变异后的序列
        sequence = list(ref_sequence)
        for pos, ref, alt in variants:
            idx = pos - start
            sequence[idx:idx + len(ref)] = list(alt)
        return ''.join(sequence), False
    except OSError:
        return None, True


def test():
    chrom_id = 0
    sample_name = 'HG00131'  # 个体名
    haplotype = 1  # 单倍体编号：0 or 1
    start = 10390  # 起始位置
    end = 10390 + 44  # 结束位置
    sequence, is_ref = query_haplotype(sample_name, haplotype, chrom_id, start, end)
    print(sequence, is_ref)


def process_reference_sampling(_):
    chrom_id = random.randint(0, 22)
    chrom = list(chromosomes.keys())[chrom_id]
    chrom_len = chromosomes_length[chrom]
    start = random.randint(1, chrom_len - MAX_LENGTH_OF_SEQ + 1)
    end = start + MAX_LENGTH_OF_SEQ - 1
    file_index = random.randint(0, NUMBER_OF_FILE - 1)
    sequence, is_ref = query_haplotype(None, None, chrom_id, start, end)
    with locks_of_write[file_index]:
        with open(f'./reference_sampling/{file_index}.txt', 'a') as file:
            file.write(sequence + '\n')


def reference_sampling():
    cur_time = time.time()
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(process_reference_sampling, range(NUMBER_OF_SEQ))
    print(time.time() - cur_time)


def process_random_sampling(_):
    sequence, is_ref = None, None
    while sequence is None:
        chrom_id = random.randint(0, 22)
        chrom = list(chromosomes.keys())[chrom_id]
        chrom_len = chromosomes_length[chrom]
        haplotype = random.randint(0, 1) if chrom != 'chrX' else 0
        start = random.randint(1, chrom_len - MAX_LENGTH_OF_SEQ + 1)
        end = start + MAX_LENGTH_OF_SEQ - 1
        with locks_of_vcf[chrom_id]:
            sample = random.choice(list(vcf[chrom].header.samples))
        sequence, is_ref = query_haplotype(sample, haplotype, chrom_id, start, end)

    file_index = random.randint(0, NUMBER_OF_FILE - 1)
    with locks_of_write[file_index]:
        with open(f'./random_sampling/{file_index}.txt', 'a') as file:
            file.write(sequence + '\n')


def random_sampling():
    cur_time = time.time()
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(process_random_sampling, range(NUMBER_OF_SEQ))
    print(time.time() - cur_time)


def process_variant_sampling(_):
    sequence, is_ref = None, True
    while is_ref:
        chrom_id = random.randint(0, 22)
        chrom = list(chromosomes.keys())[chrom_id]
        chrom_len = chromosomes_length[chrom]
        haplotype = random.randint(0, 1) if chrom != 'chrX' else 0
        start = random.randint(1, chrom_len - MAX_LENGTH_OF_SEQ + 1)
        end = start + MAX_LENGTH_OF_SEQ - 1
        with locks_of_vcf[chrom_id]:
            sample = random.choice(list(vcf[chrom].header.samples))
        sequence, is_ref = query_haplotype(sample, haplotype, chrom_id, start, end)
    file_index = random.randint(0, NUMBER_OF_FILE - 1)
    with locks_of_write[file_index]:
        with open(f'./variant_sampling/{file_index}.txt', 'a') as file:
            file.write(sequence + '\n')


def variant_sampling():
    cur_time = time.time()
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(process_variant_sampling, range(NUMBER_OF_SEQ))
    print(time.time() - cur_time)


if __name__ == '__main__':
    test()
    reference_sampling()
    random_sampling()
    variant_sampling()


