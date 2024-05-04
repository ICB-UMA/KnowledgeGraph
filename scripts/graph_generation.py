import os
import pickle
import networkx as nx
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from utils import utils
from utils.logger import setup_custom_logger  

"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

def parse_args():
    """
    Parses command line arguments for graph generation.
    
    Returns:
        argparse.Namespace: Parsed arguments with default values and help descriptions.
    """
    parser = argparse.ArgumentParser(description="Generate knowledge graphs from UMLS dataset.")
    parser.add_argument('--path', type=str, default="../../data/UMLS/2023AA/META/",
                        help="Path to the UMLS dataset directory.")
    parser.add_argument('--chunk_size', type=int, default=500000,
                        help="Chunk size for processing large files.")
    parser.add_argument('--output_path', type=str, default="utils/",
                        help="Path to save the output files like graphs and dictionaries.")
    parser.add_argument('--log_file', type=str, default="",
                        help="Path to the log file. If not provided, logging will be output to console.")
    return parser.parse_args()

def create_graphs(path, chunk_size, output_path, log_file=None):
    """
    Creates and saves knowledge graphs and dictionaries based on UMLS data.
    
    Args:
        path (str): Path where the UMLS data files are located.
        chunk_size (int): Number of records to process at a time.
        output_path (str): Base directory to save output files.
        log_file (str): path to log file
    """
    if log_file:
        logger = setup_custom_logger('graph_generation', log_file=log_file)
    else:
        logger = setup_custom_logger('graph_generation')
    
    logger.info("Starting the graph generation process...")

    # Ensure output path exists
    os.makedirs(output_path, exist_ok=True)

    colnames = utils.extract_column_names_from_ctl_file(os.path.join(path, "MRCONSO.ctl"))
    df_conso = utils.read_rrf_file_in_chunks(os.path.join(path, "MRCONSO.RRF"), chunk_size, colnames, dtype_dict={"CUI": str})
    logger.info("Processed MRCONSO.RRF")

    colnames = utils.extract_column_names_from_ctl_file(os.path.join(path, "MRHIER.ctl"))
    df_hier = utils.read_rrf_file_in_chunks(os.path.join(path, "MRHIER.RRF"), chunk_size, colnames, dtype_dict={"CUI": str})
    logger.info("Processed MRHIER.RRF, now generating graphs...")

    df_conso_sn = df_conso[df_conso['SAB'].isin(["SNOMEDCT_US", "SCTSPA"])]
    scui_to_cui_dict = df_conso_sn.groupby('SCUI')['CUI'].agg(lambda x: list(set(x))).to_dict()
    aui_to_cui_dict = df_conso_sn.set_index('AUI')['CUI'].to_dict()

    df_hier['CUI1'] = df_hier['PAUI'].map(aui_to_cui_dict).ffill()
    df_hier.rename(columns={'CUI': 'CUI2'}, inplace=True)
    df_parent_child = df_hier[['CUI1', 'CUI2']].drop_duplicates()

    G = nx.DiGraph()
    grouped = df_conso_sn.groupby('CUI')['STR'].agg(lambda x: list(set(x))).reset_index()
    for _, row in grouped.iterrows():
        G.add_node(row['CUI'], name=row['STR'])

    for _, row in df_parent_child.iterrows():
        G.add_edge(row['CUI1'], row['CUI2'])

    pickle.dump(scui_to_cui_dict, open(os.path.join(output_path, 'scui_to_cui_dict.pkl'), 'wb'))
    pickle.dump(G, open(os.path.join(output_path, 'graph_G.pkl'), 'wb'))
    logger.info("Graphs and dictionaries have been generated and saved successfully.")

if __name__ == "__main__":
    args = parse_args()
    create_graphs(args.path, args.chunk_size, args.output_path, args.log_file)
