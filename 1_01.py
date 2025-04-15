import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from pymorphy2 import MorphAnalyzer
from sklearn.metrics.pairwise import cosine_similarity

class QuestionSimilarityFinder:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.morph = MorphAnalyzer()
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

    def extract_morph_tags(self, text: str):
        return [str(self.morph.parse(word)[0].tag) for word in text.split()]

    def morph_penalty(self, tags1, tags2):
        diff = sum(1 for t1, t2 in zip(tags1, tags2) if t1 != t2)
        return diff / max(len(tags1), len(tags2), 1)

    def prepare_question_base(self, df: pd.DataFrame, question_column: str = "вопрос"):
        self.df = df.copy()
        self.df["embedding"] = self.df[question_column].apply(self.get_embedding)
        self.df["morph_tags"] = self.df[question_column].apply(self.extract_morph_tags)

    def get_top_similar(self, user_question: str, top_n: int = 5):
        user_embedding = self.get_embedding(user_question)
        user_tags = self.extract_morph_tags(user_question)

        similarities = []
        for idx, row in self.df.iterrows():
            sim = cosine_similarity(
                user_embedding.unsqueeze(0).numpy(),
                row["embedding"].unsqueeze(0).numpy()
            )[0][0]
            penalty = self.morph_penalty(user_tags, row["morph_tags"])
            final_score = sim - penalty
            similarities.append((final_score, idx))

        top_results = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_n]
        return self.df.loc[[idx for _, idx in top_results]].assign(score=[score for score, _ in top_results])
finder = QuestionSimilarityFinder(model_path)
finder.prepare_question_base(df)

# 3. Поиск
user_question = "Как оформить потребительский кредит?"
results = finder.get_top_similar(user_question)

# 4. Вывод
print("🔍 Вопрос пользователя:", user_question)
for i, row in results.iterrows():
    print(f"{i+1}. {row['вопрос']} (сходство - штраф = {row['score']:.4f})")
    print(f"Ответ: {row['ответ']}\n")
