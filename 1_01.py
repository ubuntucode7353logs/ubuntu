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
        Вычисляет эмбеддинги для текстов вопросов и категорий,
        и создает матрицы эмбеддингов для векторизированных вычислений.
        """
        self.df = df.copy()
        self.df["question_embedding"] = self.df[question_column].apply(self.get_embedding)
        self.df["category_embedding"] = self.df[category_column].apply(self.get_embedding)

        self.q_emb_matrix = np.stack(self.df["question_embedding"].apply(lambda t: t.numpy()))
        self.cat_emb_matrix = np.stack(self.df["category_embedding"].apply(lambda t: t.numpy()))

    def get_top_similar(self, user_question, top_n: int = 5):
        # Получаем эмбеддинг запроса пользователя
        user_q_emb = self.get_embedding(user_question)
        user_q_emb_np = user_q_emb.unsqueeze(0).numpy()  # форма (1, dim)

        # Векторизованное вычисление косинусного сходства между запросом и всеми вопросами
        q_sims = cosine_similarity(user_q_emb_np, self.q_emb_matrix)[0]

        # Получаем индексы топ-10 вопросов по сходству текста вопроса
        top10_indices = np.argsort(q_sims)[::-1][:10]
        top10_q_sims = q_sims[top10_indices]

        # Для топ-10 вопросов вычисляем сходство между запросом и их категориями
        top10_cat_emb = self.cat_emb_matrix[top10_indices]
        top10_c_sims = cosine_similarity(user_q_emb_np, top10_cat_emb)[0]

        # Формируем лог для топ-10 вопросов
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

        # Суммируем сходство по вопросу и по категории
        top10_df["суммарное_сходство"] = top10_df["сходство_вопрос"] + top10_df["сходство_категория"]

        # Сортируем топ-10 по суммарной схожести (от наибольшего к наименьшему)
        sorted_df = top10_df.sort_values(by="суммарное_сходство", ascending=False)

        # Выбираем финальный список: если отфильтрованных вопросов меньше top_n, возвращаем то, что есть
        final_top = sorted_df.head(top_n)

        return {
            "лог_поиска_топ_10": top10_df,
            "топ_5_суммарное_сходство": final_top
        }

# Пример использования:
if __name__ == '__main__':
    # Подготовка тестовых данных
    data = {
        "вопрос": [
            "Как оплатить заказ?",
            "Где находится офис?",
            "Как вернуть товар?",
            "Какие существуют способы доставки?",
            "Как получить скидку?",
            "Где можно забрать заказ?",
            "Как оформить возврат?",
            "Как связаться с поддержкой?",
            "Какие варианты оплаты возможны?",
            "Что делать, если товар поврежден?"
        ],
        "категория": [
            "оплата",
            "локация",
            "возврат",
            "доставка",
            "скидка",
            "доставка",
            "возврат",
            "поддержка",
            "оплата",
            "возврат"
        ]
    }
    df = pd.DataFrame(data)

    # Инициализация и подготовка базы
    model_path = "имя_или_путь_модели"  # замените на корректный путь или имя модели
    finder = QuestionSimilarityFinder(model_path)
    finder.prepare_question_base(df)

    # Поиск
    user_question = "Как оплатить мой заказ?"
    results = finder.get_top_similar(user_question, top_n=5)

    # Вывод логов
    print("Лог (топ-10 вопросов с метриками):")
    print(results["лог_поиска_топ_10"])

    print("\nФинальный топ-5 вопросов (по суммарной схожести):")
    print(results["топ_5_суммарное_сходство"])

