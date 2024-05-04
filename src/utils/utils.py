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
