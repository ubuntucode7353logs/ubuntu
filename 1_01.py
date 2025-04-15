import torch
import pandas as pd
import numpy as np
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
        """
        Предварительно вычисляет эмбеддинги для текстов вопросов и категорий.
        Создаются матрицы эмбеддингов для ускорения векторных вычислений.
        """
        self.df = df.copy()
        # Вычисляем эмбеддинги для вопросов и категорий
        self.df["question_embedding"] = self.df[question_column].apply(self.get_embedding)
        self.df["category_embedding"] = self.df[category_column].apply(self.get_embedding)

        # Создаем матрицы эмбеддингов для вопросов и категорий
        self.q_emb_matrix = np.stack(self.df["question_embedding"].apply(lambda t: t.numpy()))
        self.cat_emb_matrix = np.stack(self.df["category_embedding"].apply(lambda t: t.numpy()))

    def get_top_similar(self, user_question, top_n: int = 5):
        # Получаем эмбеддинг запроса пользователя (для вопросов и для категорий – здесь используется один и тот же запрос)
        user_q_emb = self.get_embedding(user_question)
        user_q_emb_np = user_q_emb.unsqueeze(0).numpy()  # форма (1, dim)

        # Векторизованное вычисление сходства между запросом и всеми вопросами
        q_sims = cosine_similarity(user_q_emb_np, self.q_emb_matrix)[0]  # 1D массив, длина = число вопросов

        # Получаем индексы топ-10 вопросов по сходству текста вопроса
        top10_indices = np.argsort(q_sims)[::-1][:10]
        top10_q_sims = q_sims[top10_indices]

        # Для топ-10 вопросов вычисляем сходство между запросом и их категориями
        top10_cat_emb = self.cat_emb_matrix[top10_indices]
        top10_c_sims = cosine_similarity(user_q_emb_np, top10_cat_emb)[0]

        # Составляем лог для топ-10 вопросов
        top10_logs = []
        for rank, orig_idx in enumerate(top10_indices):
            row = self.df.iloc[orig_idx]
            top10_logs.append({
                "index": orig_idx,
                "вопрос": row["вопрос"],
                "категория": row["категория"],
                "сходство_вопрос": top10_q_sims[rank],
                "сходство_категория": top10_c_sims[rank]
            })

        top10_df = pd.DataFrame(top10_logs)

        # Выбираем «выбранную» категорию – ту, у которой сходство (запрос – категория) максимально среди топ-10
        best_cat_rel_idx = np.argmax(top10_c_sims)
        best_category = top10_df.loc[best_cat_rel_idx, "категория"]
        # Получаем эмбеддинг выбранной категории (по оригинальному индексу вопроса)
        best_category_orig_idx = top10_df.loc[best_cat_rel_idx, "index"]
        best_category_emb = self.df.loc[best_category_orig_idx, "category_embedding"].numpy().reshape(1, -1)

        # Вычисляем сходство между эмбеддингами категорий топ-10 вопросов и выбранной категорией
        top10_cat_sim_to_best = cosine_similarity(best_category_emb, top10_cat_emb)[0]

        # Добавляем новую колонку с этой метрикой
        top10_df = top10_df.assign(сходство_категории_к_выбранной=top10_cat_sim_to_best)

        # Сортируем топ-10 по данной метрике (от наибольшего к наименьшему)
        sorted_df = top10_df.sort_values(by="сходство_категории_к_выбранной", ascending=False)

        # Выбираем финальный список: если отфильтрованных вопросов меньше top_n, возвращаем то, что есть
        final_top = sorted_df.head(top_n)

        return {
            "лог_поиска_топ_10": top10_df,
            "выбранная_категория": best_category,
            "топ_по_категории": final_top
        }
