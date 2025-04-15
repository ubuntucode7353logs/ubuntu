import torch
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
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
               torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    def get_embedding(self, sentence: str):
        encoded_input = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return self.mean_pooling(model_output, encoded_input['attention_mask'])[0].cpu()

    def prepare_question_base(self, df: pd.DataFrame, question_column: str = "вопрос", category_column: str = "категория"):
        self.df = df.copy()
        self.df["question_embedding"] = self.df[question_column].apply(self.get_embedding)
        self.df["category_embedding"] = self.df[category_column].apply(self.get_embedding)

    def get_top_similar(self, user_question: str, user_category: str = "", top_n: int = 5):
        # Получаем эмбеддинги для пользовательского запроса и категории
        question_emb = self.get_embedding(user_question)
        category_emb = self.get_embedding(user_category) if user_category else torch.zeros_like(question_emb)

        similarities = []
        for idx, row in self.df.iterrows():
            q_sim = cosine_similarity(
                question_emb.unsqueeze(0).numpy(),
                row["question_embedding"].unsqueeze(0).numpy()
            )[0][0]
            c_sim = cosine_similarity(
                category_emb.unsqueeze(0).numpy(),
                row["category_embedding"].unsqueeze(0).numpy()
            )[0][0] if user_category else 0
            total_sim = (q_sim + c_sim) / 2  # усреднение схожести
            similarities.append((total_sim, idx))

        # Топ 10 по общей схожести
        top_10 = sorted(similarities, key=lambda x: x[0], reverse=True)[:10]
        top_df = self.df.loc[[idx for _, idx in top_10]].assign(score=[score for score, _ in top_10])

        # Сортировка по категории
        top_df = top_df.sort_values(by="категория")

        return top_df.head(top_n)
