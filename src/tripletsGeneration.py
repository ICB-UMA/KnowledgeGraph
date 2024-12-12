import pandas as pd
import networkx as nx
from typing import List, Dict, Optional, Set

"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

class TripletsGeneration:
    """
    A base class for generating triplets.

    Attributes:
        df (pd.DataFrame): DataFrame with necessary columns.
    """

    def __init__(
        self,
        df: pd.DataFrame
    ):
        """
        Initialize the TripletsGeneration class with a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with necessary columns.
        """
        self.df = df

    def generate_triplets(self) -> pd.DataFrame:
        """
        Placeholder method to be overridden by subclasses.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class TopHardTriplets(TripletsGeneration):
    """
    Subclass of TripletsGeneration that generates top-hard triplets.
    """

    def generate_triplets(
        self,
        num_negatives: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generates top-hard triplets.

        Args:
            num_negatives (Optional[int]): Number of negative samples to consider. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing anchor, positive, and negative columns.
        """
        results = []
        for _, row in self.df.iterrows():
            term, correct_code = row['term'], row['code']
            candidate_codes, candidate_texts = row['codes'], row['candidates']

            if correct_code in candidate_codes:
                positive_index = candidate_codes.index(correct_code)
                positive_text = candidate_texts[positive_index]

                negatives = (
                    candidate_texts[:num_negatives]
                    if num_negatives else candidate_texts[:positive_index]
                )
                for neg_text in negatives:
                    results.append((term, positive_text, neg_text))

        return pd.DataFrame(results, columns=["anchor", "positive", "negative"])


class HardTripletsKG(TripletsGeneration):
    """
    Subclass of TripletsGeneration that generates hard triplets using a knowledge graph.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        G: nx.Graph,
        scui_to_cui_dict: Dict[str, List[str]],
        depth: int,
        bidirectional: bool = False
    ):
        """
        Initialize HardTripletsKG class.

        Args:
            df (pd.DataFrame): DataFrame with necessary columns.
            G (nx.Graph): Knowledge graph.
            scui_to_cui_dict (Dict[str, List[str]]): Mapping from SCUIs to CUIs.
            depth (int): Depth to explore in the knowledge graph.
            bidirectional (bool): Whether to consider bidirectional connections in the graph. Defaults to False.
        """
        super().__init__(df)
        self.G = G
        self.scui_to_cui_dict = scui_to_cui_dict
        self.depth = depth
        self.bidirectional = bidirectional

    def add_related_as_negatives(
        self,
        code: str,
        exclude_code: str,
        depth: int
    ) -> Set[str]:
        """
        Recursively add related concepts as negatives.

        Args:
            code (str): Concept code.
            exclude_code (str): Code to exclude.
            depth (int): Depth to explore in the knowledge graph.

        Returns:
            Set[str]: Set of negative concept codes.
        """
        if code == exclude_code or depth == 0 or code not in self.G:
            return set()

        negatives = {code}
        connections = (
            list(self.G.predecessors(code)) + list(self.G.successors(code))
            if self.bidirectional else list(self.G.predecessors(code))
        )

        for connection in connections:
            if connection != exclude_code:
                negatives.add(connection)
                negatives.update(self.add_related_as_negatives(connection, exclude_code, depth - 1))

        return negatives

    def generate_triplets(self) -> pd.DataFrame:
        """
        Generates hard triplets using knowledge graph.

        Returns:
            pd.DataFrame: DataFrame containing anchor, positive, and negative columns.
        """
        results = []
        for _, row in self.df.iterrows():
            term, correct_code = row['term'], row['code']
            candidate_codes, candidate_texts = row['codes'], row['candidates']

            if correct_code in candidate_codes:
                positive_index = candidate_codes.index(correct_code)
                positive_text = candidate_texts[positive_index]

                mapped_positive_code = self.scui_to_cui_dict.get(correct_code, [None])[0]
                positive_texts = {positive_text}
                if mapped_positive_code and mapped_positive_code in self.G.nodes:
                    positive_texts.update(self.G.nodes[mapped_positive_code].get('name', []))

                direct_negatives = set(candidate_texts)
                extended_negatives = set()

                for candidate_code in candidate_codes:
                    mapped_code = self.scui_to_cui_dict.get(candidate_code, [None])[0]
                    if mapped_code:
                        extended_negatives.add(mapped_code)
                        extended_negatives.update(self.add_related_as_negatives(mapped_code, correct_code, self.depth))

                extended_negative_texts = {
                    name for cui in extended_negatives
                    if cui in self.G.nodes for name in self.G.nodes[cui].get('name', [])
                }

                combined_negatives = direct_negatives | extended_negative_texts

                for pos_text in positive_texts:
                    for neg_text in combined_negatives:
                        if neg_text not in positive_texts:
                            results.append((term, pos_text, neg_text))

        return pd.DataFrame(results, columns=["anchor", "positive", "negative"])


class SimilarityHardTriplets(TripletsGeneration):
    """
    Subclass of TripletsGeneration that generates triplets based on text similarity.
    """

    def generate_triplets(
        self,
        similarity_threshold: float
    ) -> pd.DataFrame:
        """
        Generates triplets based on text similarity.

        Args:
            similarity_threshold (float): Similarity threshold.

        Returns:
            pd.DataFrame: DataFrame containing anchor, positive, and negative columns.
        """
        results = []
        for _, row in self.df.iterrows():
            term, correct_code = row['term'], row['code']
            candidate_texts, candidate_similarities = row['candidates'], row['similarities']

            if correct_code in row['codes']:
                positive_index = row['codes'].index(correct_code)
                positive_text = candidate_texts[positive_index]

                negative_indices = [
                    i for i, sim in enumerate(candidate_similarities)
                    if sim > similarity_threshold and i != positive_index
                ]
                for i in negative_indices:
                    results.append((term, positive_text, candidate_texts[i]))

        return pd.DataFrame(results, columns=["anchor", "positive", "negative"])
