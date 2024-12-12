import os
import sys
import pickle
import networkx as nx
import argparse
import pandas as pd
from typing import Dict, List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from utils import *
from logger import setup_custom_logger

"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for graph generation.

    Returns:
        argparse.Namespace: Parsed arguments with default values and help descriptions.
    """
    parser = argparse.ArgumentParser(description="Generate knowledge graphs from UMLS dataset.")
    parser.add_argument(
        '--path',
        type=str,
        default="../../../data/UMLS/2023AA/META/",
        help="Path to the UMLS dataset directory."
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=500000,
        help="Chunk size for processing large files."
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default="utils/",
        help="Path to save the output files like graphs and dictionaries."
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default="",
        help="Path to the log file. If not provided, logging will be output to console."
    )
    return parser.parse_args()


def create_graphs(
    path: str,
    chunk_size: int,
    output_path: str,
    log_file: Optional[str] = None
) -> None:
    """
    Create and save knowledge graphs and dictionaries based on UMLS data.

    Args:
        path (str): Path where the UMLS data files are located.
        chunk_size (int): Number of records to process at a time.
        output_path (str): Base directory to save output files.
        log_file (Optional[str]): Path to the log file. If not provided, logs are output to console.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    logger = setup_custom_logger('graph_generation', log_file=log_file) if log_file else setup_custom_logger('graph_generation')
    logger.info("Starting the graph generation process...")

    # Ensure output path exists
    os.makedirs(output_path, exist_ok=True)

    # Process MRCONSO
    mrconso_ctl = os.path.join(path, "MRCONSO.ctl")
    mrconso_rrf = os.path.join(path, "MRCONSO.RRF")
    if not os.path.exists(mrconso_ctl) or not os.path.exists(mrconso_rrf):
        raise FileNotFoundError(f"Missing MRCONSO files: {mrconso_ctl} or {mrconso_rrf}")

    colnames = extract_column_names_from_ctl_file(mrconso_ctl)
    df_conso = read_rrf_file_in_chunks(mrconso_rrf, chunk_size, colnames, dtype_dict={"CUI": str})
    logger.info("Processed MRCONSO.RRF")

    # Process MRHIER
    mrhier_ctl = os.path.join(path, "MRHIER.ctl")
    mrhier_rrf = os.path.join(path, "MRHIER.RRF")
    if not os.path.exists(mrhier_ctl) or not os.path.exists(mrhier_rrf):
        raise FileNotFoundError(f"Missing MRHIER files: {mrhier_ctl} or {mrhier_rrf}")

    colnames = extract_column_names_from_ctl_file(mrhier_ctl)
    df_hier = read_rrf_file_in_chunks(mrhier_rrf, chunk_size, colnames, dtype_dict={"CUI": str})
    logger.info("Processed MRHIER.RRF, now generating graphs...")

    # Filter SNOMED CT terms and create mappings
    df_conso_sn = df_conso[df_conso['SAB'].isin(["SNOMEDCT_US", "SCTSPA"])]
    scui_to_cui_dict = df_conso_sn.groupby('SCUI')['CUI'].agg(lambda x: list(set(x))).to_dict()
    aui_to_cui_dict = df_conso_sn.set_index('AUI')['CUI'].to_dict()

    # Build hierarchical relationships
    df_hier['CUI1'] = df_hier['PAUI'].map(aui_to_cui_dict).ffill()
    df_hier.rename(columns={'CUI': 'CUI2'}, inplace=True)
    df_parent_child = df_hier[['CUI1', 'CUI2']].drop_duplicates()

    # Create graph
    G = nx.DiGraph()
    grouped = df_conso_sn.groupby('CUI')['STR'].agg(lambda x: list(set(x))).reset_index()

    logger.info("Adding nodes to the graph...")
    for _, row in grouped.iterrows():
        G.add_node(row['CUI'], name=row['STR'])

    logger.info("Adding edges to the graph...")
    for _, row in df_parent_child.iterrows():
        if pd.notnull(row['CUI1']) and pd.notnull(row['CUI2']):
            G.add_edge(row['CUI1'], row['CUI2'])

    # Save graph and mappings
    scui_dict_path = os.path.join(output_path, 'scui_to_cui_dict.pkl')
    graph_path = os.path.join(output_path, 'graph_G.pkl')

    with open(scui_dict_path, 'wb') as scui_file:
        pickle.dump(scui_to_cui_dict, scui_file)
    logger.info(f"SCUI-to-CUI dictionary saved to {scui_dict_path}")

    with open(graph_path, 'wb') as graph_file:
        pickle.dump(G, graph_file)
    logger.info(f"Graph saved to {graph_path}")

    logger.info("Graphs and dictionaries have been generated and saved successfully.")


if __name__ == "__main__":
    args = parse_args()
    create_graphs(args.path, args.chunk_size, args.output_path, args.log_file)
