import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np
import faiss
import pandas as pd

class FaissEncoder:
    def __init__(self, MODEL_NAME: str, F_TYPE: str, MAX_LENGTH: int, vocab: pd.DataFrame):
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.f_type = F_TYPE
        self.vocab = vocab
        self.arr_text = self.vocab['term'].values.tolist()
        self.arr_codes = self.vocab['code'].values.tolist()
        self.arr_text_id = np.arange(len(self.vocab))

        # Setup device and DataParallel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

    def encode(self, texts, batch_size):
        all_embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        for batch_idx in tqdm(range(num_batches), desc="Encoding"):
            batch_texts = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
                all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def fitFaiss(self, batch_size=32):
        embeddings = self.encode(self.arr_text, batch_size)
        embeddings = embeddings.astype('float32')
        index_type = faiss.IndexFlatL2 if self.f_type == "FlatL2" else faiss.IndexFlatIP
        faiss_index = faiss.IndexIDMap(index_type(embeddings.shape[1]))

        if self.f_type == "FlatIP":
            faiss.normalize_L2(embeddings)
        
        faiss_index.add_with_ids(embeddings, self.arr_text_id)
        self.faiss_index = faiss_index

    def getCandidates(self, texts, k=200, batch_size=64):
        encoded_entities = self.encode(texts, batch_size).astype('float32')
        if self.f_type == "FlatIP":
            faiss.normalize_L2(encoded_entities)

        sim, indexes = self.faiss_index.search(encoded_entities, k * 3)
        return self._process_results(indexes, sim, k)

    def _process_results(self, indexes, sim, k):
        candidates, candidates_codes, candidates_sims = [], [], []
        for idx_list, sim_list in zip(indexes, sim):
            seen_codes, unique_candidates, unique_codes, unique_sims = set(), [], [], []
            for idx, sim_score in zip(idx_list, sim_list):
                if idx < len(self.arr_text) and self.arr_codes[idx] not in seen_codes:
                    seen_codes.add(self.arr_codes[idx])
                    unique_candidates.append(self.arr_text[idx])
                    unique_codes.append(self.arr_codes[idx])
                    unique_sims.append(sim_score)
                    if len(unique_candidates) == k:
                        break
            candidates.append(unique_candidates)
            candidates_codes.append(unique_codes)
            candidates_sims.append(unique_sims)
        return candidates, candidates_codes, candidates_sims

    def evaluate(self, eval_df, k_values, batch_size=64):
        _, candidates_codes, _ = self.getCandidates(eval_df['term'].tolist(), max(k_values) + 150, batch_size)
        precision_at_k = {}
        for k in k_values:
            correct_predictions = sum(1 for true_code, candidates in zip(eval_df['code'], candidates_codes) if true_code in candidates[:k])
            precision_at_k[k] = correct_predictions / len(eval_df)
        return precision_at_k