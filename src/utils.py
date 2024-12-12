from tqdm.auto import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Optional

"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

def extract_column_names_from_ctl_file(
    ctl_file_path: str) -> List[str]:
    """
    Extract column names from a CTL file.

    Args:
        ctl_file_path (str): Path to the CTL file.

    Returns:
        List[str]: List of column names extracted from the file.
    """
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


def read_rrf_file_in_chunks(
    file_path: str, 
    chunk_size: int, 
    columns: List[str], 
    dtype_dict: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Read an RRF file in chunks and concatenate the chunks into a DataFrame.

    Args:
        file_path (str): Path to the RRF file.
        chunk_size (int): Number of lines per chunk.
        columns (List[str]): List of column names.
        dtype_dict (Optional[Dict[str, str]]): Dictionary specifying data types for columns.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing all chunks.
    """
    chunk_list = []

    with tqdm(desc="Processing", unit="line") as pbar:
        for chunk in pd.read_csv(
            file_path,
            sep='|',
            chunksize=chunk_size,
            na_filter=False,
            low_memory=True,  
            dtype=dtype_dict,
            usecols=range(len(columns)),  
            names=columns,  
        ):
            chunk_list.append(chunk)
            pbar.update(len(chunk))

    df = pd.concat(chunk_list, ignore_index=True)
    return df

def load_corpus_data(
    base_path: str, 
    corpus: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load test, train, and gazetteer data for a specific corpus.

    Args:
        base_path (str): Base path containing the corpus directories.
        corpus (str): Name of the corpus ("SympTEMIST", "MedProcNER", or "DisTEMIST").

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Test, train, and gazetteer DataFrames.
    
    Raises:
        ValueError: If the corpus is not supported.
    """
    corpus_paths = {
        "SympTEMIST": {
            "test": f"{base_path}/SympTEMIST/symptemist-complete_240208/symptemist_test/subtask2-linking/symptemist_tsv_test_subtask2.tsv",
            "train": f"{base_path}/SympTEMIST/symptemist-complete_240208/symptemist_train/subtask2-linking/symptemist_tsv_train_subtask2_complete.tsv",
            "gaz": f"{base_path}/SympTEMIST/symptemist-complete_240208/symptemist_gazetteer/symptemist_gazetter_snomed_ES_v2.tsv"
        },
        "MedProcNER": {
            "test": f"{base_path}/MedProcNER/medprocner_gs_train+test+gazz+multilingual+crossmap_230808/medprocner_test/tsv/medprocner_tsv_test_subtask2.tsv",
            "train": f"{base_path}/MedProcNER/medprocner_gs_train+test+gazz+multilingual+crossmap_230808/medprocner_train/tsv/medprocner_tsv_train_subtask2.tsv",
            "gaz": f"{base_path}/MedProcNER/medprocner_gs_train+test+gazz+multilingual+crossmap_230808/medprocner_gazetteer/gazzeteer_medprocner_v1_noambiguity.tsv"
        },
        "DisTEMIST": {
            "test": f"{base_path}/DisTEMIST/distemist_zenodo/test_annotated/subtrack2_linking/distemist_subtrack2_test_linking.tsv",
            "train": f"{base_path}/DisTEMIST/distemist_zenodo/training/subtrack2_linking/distemist_subtrack2_training2_linking.tsv",
            "gaz": f"{base_path}/DisTEMIST/dictionary_distemist.tsv"
        }
    }

    if corpus not in corpus_paths:
        raise ValueError(f"Unsupported corpus: {corpus}")

    paths = corpus_paths[corpus]
    test_df = pd.read_csv(paths["test"], sep="\t", dtype={"code": str})
    train_df = pd.read_csv(paths["train"], sep="\t", dtype={"code": str})
    df_gaz = pd.read_csv(paths["gaz"], sep="\t", dtype={"code": str})

    if corpus == "SympTEMIST" or corpus == "MedProcNER":
        test_df.rename(columns={'text': 'term'}, inplace=True)
        train_df.rename(columns={'text': 'term'}, inplace=True)
    elif corpus == "DisTEMIST":
        test_df.rename(columns={'span': 'term'}, inplace=True)
        train_df.rename(columns={'span': 'term'}, inplace=True)

    return test_df, train_df, df_gaz
