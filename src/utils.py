from tqdm.auto import tqdm
import pandas as pd

"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

def extract_column_names_from_ctl_file(ctl_file_path):
    with open(ctl_file_path, 'r') as file:
        lines = file.readlines()

    # Find the index of the line containing 'trailing nullcols'
    start_index = next(i for i, line in enumerate(lines) if 'trailing nullcols' in line) + 1
    # The second last line of the file is the boundary for processing
    end_index = len(lines) - 1

    column_names = []
    for line in lines[start_index:end_index]:
        # Extract only the column name, assuming it is the first word
        name = line.split()[0].replace('(', '').replace(')', '').strip()
        if name:
            column_names.append(name)

    return column_names


def read_rrf_file_in_chunks(file_path, chunk_size, columns, dtype_dict=None):
    total_lines = sum(1 for line in open(file_path, 'r', encoding='utf8'))
    chunk_list = []

    with tqdm(total=total_lines, desc="Processing", unit="line") as pbar:
        for chunk in pd.read_csv(file_path, sep='|', chunksize=chunk_size, na_filter=False, low_memory=False, dtype=dtype_dict):
            chunk = chunk.iloc[:, :len(columns)]
            chunk_list.append(chunk)
            pbar.update(min(chunk_size, total_lines - pbar.n))

    df = pd.concat(chunk_list, axis=0)
    df.columns = columns
    return df

def load_corpus_data(base_path, corpus):
    """Load testing data and gazetteer based on the specified corpus and base path."""
    if corpus == "SympTEMIST":
        test_df = pd.read_csv(f"{base_path}/SympTEMIST/symptemist-complete_240208/symptemist_test/subtask2-linking/symptemist_tsv_test_subtask2.tsv", sep="\t", header=0, dtype={"code": str})
        test_df = test_df.rename(columns={'text': 'term'})
        df_gaz = pd.read_csv(f"{base_path}/SympTEMIST/symptemist-complete_240208/symptemist_gazetteer/symptemist_gazetter_snomed_ES_v2.tsv", sep="\t", header=0, dtype={"code": str})
        train_df = pd.read_csv(f"{base_path}/SympTEMIST/symptemist-complete_240208/symptemist_train/subtask2-linking/symptemist_tsv_train_subtask2_complete.tsv", sep="\t", header=0, dtype={"code": str})
        train_df = train_df.rename(columns={'text': 'term'})
    elif corpus == "MedProcNER":
        test_df = pd.read_csv(f"{base_path}/MedProcNER/medprocner_gs_train+test+gazz+multilingual+crossmap_230808/medprocner_test/tsv/medprocner_tsv_test_subtask2.tsv", sep="\t", header=0, dtype={"code": str})
        test_df = test_df.rename(columns={'text': 'term'})
        df_gaz = pd.read_csv(f"{base_path}/MedProcNER/medprocner_gs_train+test+gazz+multilingual+crossmap_230808/medprocner_gazetteer/gazzeteer_medprocner_v1_noambiguity.tsv", sep="\t", header=0, dtype={"code": str})
        train_df = pd.read_csv(f"{base_path}/MedProcNER/medprocner_gs_train+test+gazz+multilingual+crossmap_230808/medprocner_train/tsv/medprocner_tsv_train_subtask2.tsv", sep="\t", header=0, dtype={"code": str})
        train_df = train_df.rename(columns={'text': 'term'})
    elif corpus == "DisTEMIST":
        test_df = pd.read_csv(f"{base_path}/DisTEMIST/distemist_zenodo/test_annotated/subtrack2_linking/distemist_subtrack2_test_linking.tsv", sep="\t", header=0, dtype={"code": str})
        test_df = test_df.rename(columns={'span': 'term'})
        df_gaz = pd.read_csv(f"{base_path}/DisTEMIST/dictionary_distemist.tsv", sep="\t", header=0, dtype={"code": str})
        train_df = pd.read_csv(f"{base_path}/DisTEMIST/distemist_zenodo/training/subtrack2_linking/distemist_subtrack2_training2_linking.tsv", sep="\t", header=0, dtype={"code": str})
        train_df = train_df.rename(columns={'span': 'term'})
    else:
        raise ValueError(f"Unsupported corpus: {corpus}")
    
    return test_df, train_df, df_gaz
