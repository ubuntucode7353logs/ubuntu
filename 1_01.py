import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class QuestionSimilarityFinder:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # эмбеддинги токенов
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
               torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    def get_embedding(self, sentence: str):
        encoded_input = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return self.mean_pooling(model_output, encoded_input['attention_mask'])[0].cpu()

    def prepare_question_base(self, df: pd.DataFrame, question_column: str = "вопрос"):
        self.df = df.copy()
        self.df["embedding"] = self.df[question_column].apply(self.get_embedding)

    def get_top_similar(self, user_question: str, top_n: int = 5):
        user_embedding = self.get_embedding(user_question)

        similarities = []
        for idx, row in self.df.iterrows():
            sim = cosine_similarity(
                user_embedding.unsqueeze(0).numpy(),
                row["embedding"].unsqueeze(0).numpy()
            )[0][0]
            similarities.append((sim, idx))

        top_results = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_n]
        return self.df.loc[[idx for _, idx in top_results]].assign(score=[score for score, _ in top_results])
