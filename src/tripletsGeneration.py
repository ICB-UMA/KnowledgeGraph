import pandas as pd

"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

class TripletsGeneration:
    """
    A base class for generating triplets.

    Attributes:
        df (pandas.DataFrame): DataFrame with necessary columns.
    """

    def __init__(self, df):
        """
        Initialize the TripletsGeneration class with a DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame with necessary columns.
        """
        self.df = df

    def generate_triplets(self):
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

    def generate_triplets(self, num_negatives=None):
        """
        Generates top-hard triplets.

        Args:
            num_negatives (int, optional): Number of negative samples to consider. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame containing anchor, positive, and negative columns.
        """
        results = []
        for index, row in self.df.iterrows():
            term, correct_code = row['term'], row['code']
            candidate_codes, candidate_texts = row['codes'], row['candidates'] 

            if correct_code in candidate_codes:
                positive_index = candidate_codes.index(correct_code)
                positive_text = candidate_texts[positive_index]

                negatives = candidate_texts[:num_negatives] if num_negatives else candidate_texts[:positive_index]
                for neg_text in negatives:
                    results.append((term, positive_text, neg_text))

        return pd.DataFrame(results, columns=["anchor", "positive", "negative"])

class HardTripletsKG(TripletsGeneration):
    """
    Subclass of TripletsGeneration that generates hard triplets using knowledge graph.
    """

    def __init__(self, df, G, scui_to_cui_dict, depth, bidirectional=False):
        """
        Initialize HardTripletsKG class.

        Args:
            df (pandas.DataFrame): DataFrame with necessary columns.
            G (networkx.Graph): Knowledge graph.
            scui_to_cui_dict (dict): Mapping from source concept unique identifiers (SCUIs) to concept unique identifiers (CUIs).
            depth (int): Depth to explore in the knowledge graph.
            bidirectional (bool, optional): Whether to consider bidirectional connections in the knowledge graph. Defaults to False.
        """
        super().__init__(df)
        self.G = G
        self.scui_to_cui_dict = scui_to_cui_dict
        self.depth = depth
        self.bidirectional = bidirectional

    def add_related_as_negatives(self, code, exclude_code, depth, G, bidirectional):
        """
        Recursively add related concepts as negatives.

        Args:
            code (str): Concept code.
            exclude_code (str): Code to exclude.
            depth (int): Depth to explore in the knowledge graph.
            G (networkx.Graph): Knowledge graph.
            bidirectional (bool): Whether to consider bidirectional connections in the knowledge graph.

        Returns:
            list: List of negative concept codes.
        """
        if code == exclude_code or depth == 0 or code not in G:
            return []
        
        negatives = [code]
        if bidirectional:
            connections = list(G.predecessors(code)) + list(G.successors(code))
        else:
            connections = list(G.predecessors(code))
        
        for connection in connections:
            if connection != exclude_code:
                negatives.append(connection)
                negatives.extend(self.add_related_as_negatives(connection, exclude_code, depth - 1, G, bidirectional))

        return list(set(negatives))  # Return unique items

    def generate_triplets(self):
        """
        Generates hard triplets using knowledge graph.

        Returns:
            pandas.DataFrame: DataFrame containing anchor, positive, and negative columns.
        """
        results = []
        for idx, row in self.df.iterrows():
            term = row['term']
            correct_code = row['code']
            candidate_codes = row['codes']
            candidate_texts = row['candidates']

            if correct_code in candidate_codes:
                positive_index = candidate_codes.index(correct_code)
                positive_text = candidate_texts[positive_index]

                mapped_positive_code = self.scui_to_cui_dict.get(correct_code, [None])[0]
                positive_texts = {positive_text}
                if mapped_positive_code in self.G.nodes:
                    positive_texts.update(self.G.nodes[mapped_positive_code].get('name', []))
                
                direct_negatives_texts = candidate_texts[:]  
                extended_negatives = []

                for neg_code in candidate_codes:
                    mapped_neg_code = self.scui_to_cui_dict.get(neg_code, [None])[0]
                    if mapped_neg_code:
                        extended_negatives.append(mapped_neg_code)
                        extended_negatives.extend(self.add_related_as_negatives(mapped_neg_code, correct_code, self.depth, self.G, self.bidirectional))

                extended_negatives = list(set(extended_negatives))
                extended_negative_texts = [name for cui in extended_negatives if cui in self.G.nodes for name in self.G.nodes[cui].get('name', [])]
                combined_negatives = list(set(direct_negatives_texts + extended_negative_texts))

                for pos_text in positive_texts:
                    for neg_text in combined_negatives:
                        if neg_text not in positive_texts:
                            results.append((term, pos_text, neg_text))

        return pd.DataFrame(results, columns=["anchor", "positive", "negative"])

class SimilarityHardTriplets(TripletsGeneration):
    """
    Subclass of TripletsGeneration that generates triplets based on text similarity.
    """

    def generate_triplets(self, similarity_threshold):
        """
        Generates triplets based on text similarity.

        Args:
            similarity_threshold (float): Similarity threshold.

        Returns:
            pandas.DataFrame: DataFrame containing anchor, positive, and negative columns.
        """
        results = []
        for index, row in self.df.iterrows():
            term, correct_code = row['term'], row['code']
            candidate_texts, candidate_similarities = row['candidates'], row['similarities']

            if correct_code in row['codes']:
                positive_index = row['codes'].index(correct_code)
                positive_text = candidate_texts[positive_index]

                negative_indices = [i for i, sim in enumerate(candidate_similarities) if sim > similarity_threshold and i != positive_index]
                for i in negative_indices:
                    results.append((term, positive_text, candidate_texts[i]))

        return pd.DataFrame(results, columns=["anchor", "positive", "negative"])