import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pymorphy3
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
def lemmatize_text(text):
        morph = pymorphy3.MorphAnalyzer()
        words = text.split()
        lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
        return " ".join(lemmatized_words)

class QuestionSimilarityFinderLemmatized:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.morph = pymorphy3.MorphAnalyzer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
               torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    def get_embedding(self, sentence: str):
        lemmatized_sentence = lemmatize_text(sentence)
        encoded_input = self.tokenizer(lemmatized_sentence, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return self.mean_pooling(model_output, encoded_input['attention_mask'])[0].cpu()

    def prepare_question_base(self, df, question_column="вопрос", category_column="категория"):
        self.df = df.copy()
        self.df["question_embedding"] = self.df[question_column].apply(self.get_embedding)

    def get_top_similar(self, user_question, top_n: int = 5):
        user_q_emb = self.get_embedding(user_question)

        logs = []
        for idx, row in self.df.iterrows():
            q_sim = cosine_similarity(user_q_emb.unsqueeze(0).numpy(), row["question_embedding"].unsqueeze(0).numpy())[0][0]
            logs.append({"index": idx, "вопрос": row["вопрос"], "сходство_вопрос": q_sim, "ответ": row["ответ"]})

        sim_df = pd.DataFrame(logs).sort_values(by="сходство_вопрос", ascending=False).reset_index(drop=True)
        top_5_df = sim_df.head(top_n)

        return {
            "топ_5_общая_сходство": top_5_df
        }
def get_top_similar(self, user_question, top_n: int = 5, threshold: float = 0.75):
        # 1. Получаем эмбеддинг запроса пользователя
        user_q_emb = self.get_embedding(user_question)
        user_q_emb_np = user_q_emb.unsqueeze(0).numpy()  # shape: (1, dim)

        # 2. Вычисляем сходство между запросом и всеми вопросами (по тексту)
        q_sims = cosine_similarity(user_q_emb_np, self.q_emb_matrix)[0]  # 1D массив длины = число вопросов

        # 3. Выбираем топ-10 вопросов по сходству текста
        num_top = min(len(q_sims), 10)
        top10_indices = np.argsort(q_sims)[::-1][:num_top]
        top10_q_sims = q_sims[top10_indices]

        # 4. Для топ-10 вопросов вычисляем сходство между запросом и их категориями
        top10_cat_emb = self.cat_emb_matrix[top10_indices]
        top10_c_sims = cosine_similarity(user_q_emb_np, top10_cat_emb)[0]

        # Составляем DataFrame с метриками для топ-10 вопросов
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

        # 5. Определяем преобладающую категорию среди топ-10:
        # Группируем по категории: считаем количество и среднее сходство (с запросом)
        grp = top10_df.groupby("категория").agg(
            count=('категория', 'count'),
            avg_cat_sim=('сходство_категория', 'mean')
        )
        # Комбинированная метрика: можно взять произведение количества и среднего сходства
        grp["combined"] = grp["count"] * grp["avg_cat_sim"]
        best_category = grp["combined"].idxmax()

        # 6. Вычисляем embedding для выбранной категории как среднее для всех вопросов топ-10 этой категории
        indices_for_best = top10_df[top10_df["категория"] == best_category]["index"].values
        embeddings = np.stack([self.df.loc[idx, "category_embedding"].numpy() for idx in indices_for_best])
        best_category_emb = np.mean(embeddings, axis=0).reshape(1, -1)

        # 7. Вычисляем косинусное сходство между embedding’ом каждой категории из топ-10 и выбранной категорией
        top10_cat_sim_to_best = cosine_similarity(best_category_emb, top10_cat_emb)[0]
        top10_df = top10_df.assign(сходство_категории_к_выбранной=top10_cat_sim_to_best)

        # 8. Фильтруем вопросы: оставляем те, у которых сходство между их категорией и выбранной >= threshold
        filtered_df = top10_df[top10_df["сходство_категории_к_выбранной"] >= threshold].copy()

        # 9. Если после фильтрации осталось меньше, чем top_n, возвращаем имеющиеся (иначе берём top_n по сходству текста)
        final_top = filtered_df.sort_values(by="сходство_вопрос", ascending=False).head(top_n)
        
        return {
            "лог_поиска_топ_10": top10_df,
            "преобладающая_категория": best_category,
            "отфильтрованные_вопросы": filtered_df,
            "топ_по_категории": final_top
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

from transformers import AutoModel, AutoTokenizer
import torch

# Очистим GPU, если есть
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Загрузка модели
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Перевод на GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Создание тензора
inputs = tokenizer("Пример текста", return_tensors="pt").to(device)
with torch.no_grad():
    _ = model(**inputs)

# Печать использования
if torch.cuda.is_available():
    print(f"Пиковое использование памяти: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")

