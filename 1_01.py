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

    def prepare_question_base(self, df, question_column="вопрос", category_column="категория"):
        self.df = df.copy()
        self.df["question_embedding"] = self.df[question_column].apply(self.get_embedding)
        self.df["category_embedding"] = self.df[category_column].apply(self.get_embedding)

    def get_top_similar(self, user_question, top_n: int = 5):
        user_q_emb = self.get_embedding(user_question)

        logs = []
        for idx, row in self.df.iterrows():
            q_sim = cosine_similarity(user_q_emb.unsqueeze(0).numpy(),row["question_embedding"].unsqueeze(0).numpy())[0][0]
            c_sim = cosine_similarity(user_q_emb.unsqueeze(0).numpy(),row["category_embedding"].unsqueeze(0).numpy())[0][0]
            avg_sim = (q_sim + c_sim) / 2
            logs.append({"index": idx, "вопрос": row["вопрос"], "категория": row["категория"],
                "сходство_вопрос": q_sim, "сходство_категория": c_sim, "сходство_среднее": avg_sim})

        sim_df = pd.DataFrame(logs).sort_values(by="сходство_среднее", ascending=False).reset_index(drop=True)
        top_10_df = sim_df.head(10)
        top_5_combined = top_10_df.head(top_n)

        return {
            "лог_поиска": top_10_df,
            "топ_5_общая_сходство": top_5_combined
        }
