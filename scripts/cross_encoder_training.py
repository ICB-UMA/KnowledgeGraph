import os
import argparse
import pandas as pd
import pickle
import sys

# Custom module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import faissEncoder as faiss_enc
from crossEncoder import CrossEncoderReranker
from tripletsGeneration import HardTripletsKG, SimilarityHardTriplets, TopHardTriplets
from utils.logger import setup_custom_logger

def parse_args():
    """
    Parse command line arguments.
    Returns:
        Namespace of command line arguments
    """
    parser = argparse.ArgumentParser(description="Train a cross-encoder model for entity linking.")
    parser.add_argument('--model_mapping_file', type=str, required=True, help='Path to the model mapping file')
    parser.add_argument('--corpus', type=str, default='MedProcNER', help='Name of the corpus to process')
    parser.add_argument('--corpus_path', type=str, default='../../EntityLinking/data/MedProcNER/processed_data/', help='Path to the corpus data')
    parser.add_argument('--model_path', type=str, default='../../models/spanish_sapbert_models/sapbert_15_grandparents_1epoch/', help='Model path for FAISS encoder and cross encoder')
    parser.add_argument('--hard_triplets_type', type=str, choices=['kg', 'top', 'sim', 'bkg'], default='kg', help='Type of hard triplets to generate')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length for the model')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--candidates', type=int, default=200, help='Number of candidates to generate')
    parser.add_argument('--num_negatives', type=int, default=200, help='Number of negative samples')
    parser.add_argument('--depth', type=int, default=1, help='Depth for KG triplets generation')
    parser.add_argument('--f_type', type=str, default='FlatIP', help='FAISS index type')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Optimizer weight decay')
    parser.add_argument('--eval_steps', type=int, default=250000, help='Evaluation steps')
    parser.add_argument('--test_size', default=None, type=float, help='Fraction of the data to be used as test set (optional)')
    parser.add_argument('--log_file', type=str, default=None, help='File to log to (defaults to console if not provided)')
    return parser.parse_args()


def load_data(args, logger):
    """
    Load datasets needed for training and testing.
    Args:
        args: Parsed command line arguments
        logger: Configured logger object
    Returns:
        Various pandas DataFrames required for training/testing
    """
    logger.info("Loading data...")
    df_gaz = pd.read_csv(os.path.join(args.corpus_path, "gazetteer_term_code.tsv"), sep="\t", header=0, dtype={"code": str}, low_memory=False)
    df_train_link = pd.read_csv(os.path.join(args.corpus_path, "df_link_train.tsv"), sep="\t", header=0, dtype={"code": str}, low_memory=False)
    return df_gaz, df_train_link

def prepare_model(args, df_gaz, df_train_link, logger):
    """
    Prepare and return the FAISS encoder and initial candidate generation.
    Args:
        args: Parsed command line arguments
        df_gaz: Gazetteer DataFrame
        df_train_link: Training link DataFrame
        logger: Configured logger object
    Returns:
        Initialized FAISS encoder and updated df_train_link with candidates
    """
    logger.info("Preparing model...")
    faiss_encoder = faiss_enc.FaissEncoder(args.model_path, args.f_type, args.max_length, df_gaz)
    faiss_encoder.fitFaiss()
    candidates, codes, similarities = faiss_encoder.getCandidates(df_train_link["term"].tolist(), args.candidates, args.max_length)
    df_train_link["candidates"], df_train_link["codes"], df_train_link["similarities"] = candidates, codes, similarities

def generate_triplets(args, df_train_link, logger):
    """
    Generate hard triplets based on specified method in the arguments.
    Args:
        args: Parsed command line arguments
        df_train_link: DataFrame with training data links
        logger: Configured logger object
    Returns:
        DataFrame containing generated hard triplets
    """
    logger.info("Generating triplets...")
    if args.hard_triplets_type == 'top':
        return TopHardTriplets(df_train_link).generate_triplets(args.num_negatives)
    elif args.hard_triplets_type == 'sim':
        return SimilarityHardTriplets(df_train_link).generate_triplets(similarity_threshold=0.35)
    elif args.hard_triplets_type in ['kg', 'bkg']:
        with open("../src/utils/graph_G.pkl", "rb") as f:
            G = pickle.load(f)
        with open("../src/utils/scui_to_cui_dict.pkl", "rb") as handle:
            mapping_dict = pickle.load(handle)
        return HardTripletsKG(df_train_link, G, mapping_dict, args.depth, bidirectional=(args.hard_triplets_type == 'bkg')).generate_triplets()
    else:
        logger.error("Unsupported triplets type")
        raise ValueError("Unsupported triplets type")

def train_cross_encoder(args, df_hard_triplets, logger):
    """
    Train the cross-encoder model using the generated hard triplets and save the model.
    Args:
        args: Parsed command line arguments
        df_hard_triplets: DataFrame containing the hard triplets for training
        logger: Configured logger object
    """
    logger.info("Training cross-encoder...")
    cross_encoder = CrossEncoderReranker(args.model_path, model_type="mask", max_seq_length=args.max_length)
    output_path = os.path.join("../models/", f"cef_{args.corpus.lower()}_{args.hard_triplets_type}_{args.depth}_cand_{args.num_negatives}_epoch_{args.epochs}_bs_{args.batch_size}")
    cross_encoder.train(df_hard_triplets, output_path, args.batch_size, args.epochs, optimizer_parameters={"lr": args.lr}, weight_decay=args.weight_decay, evaluation_steps=args.eval_steps, save_best_model=False, test_size=args.test_size)
    cross_encoder.save(output_path)
    logger.info(f"Model saved to {output_path}")

def main():
    args = parse_args()
    logger = setup_custom_logger('crossEncoderTraining', log_file=args.log_file)
    logger.info("Starting the training process...")
    df_gaz, df_train_link = load_data(args, logger)
    prepare_model(args, df_gaz, df_train_link, logger)
    df_hard_triplets = generate_triplets(args, df_train_link, logger)
    train_cross_encoder(args, df_hard_triplets, logger)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
