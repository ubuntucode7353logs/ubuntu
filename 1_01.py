import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. Инициализация модели и токенизатора
# -------------------------------
model_path = "имя_или_путь_модели"  # укажите нужный путь к модели
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# -------------------------------
# 2. Функции для получения эмбеддингов
# -------------------------------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # размеры: (batch_size, tokens, hidden_dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
           torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

def get_embedding(sentence: str):
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask'])[0].cpu()

# -------------------------------
# 3. Подготовка датафрейма с вопросами и категориями
# -------------------------------
# Пример датафрейма; замените на свои данные
data = {
    "вопрос": [
        "Как оплатить заказ?",
        "Где находится офис?",
        "Как вернуть товар?",
        "Какие способы доставки существуют?",
        "Как получить скидку?",
        "Где можно забрать заказ?"
    ],
    "категория": [
        "оплата",
        "локация",
        "возврат",
        "доставка",
        "скидка",
        "доставка"
    ]
}
df = pd.DataFrame(data)

# -------------------------------
# 4. Вычисляем эмбеддинги для вопросов и категорий
# -------------------------------
print("Вычисляем эмбеддинги для вопросов и категорий...")
df["question_embedding"] = df["вопрос"].apply(get_embedding)
df["category_embedding"] = df["категория"].apply(get_embedding)

# Для ускорения расчетов создаем матрицы эмбеддингов в формате numpy
q_emb_matrix = np.stack(df["question_embedding"].apply(lambda t: t.numpy()))
cat_emb_matrix = np.stack(df["category_embedding"].apply(lambda t: t.numpy()))

# -------------------------------
# 5. Вычисляем эмбеддинг запроса пользователя
# -------------------------------
user_question = "Как оплатить заказ?"
user_q_emb = get_embedding(user_question)
user_q_emb_np = user_q_emb.unsqueeze(0).numpy()

# -------------------------------
# 6. Векторизованное вычисление косинусного сходства между запросом и всеми вопросами
# -------------------------------
q_sims = cosine_similarity(user_q_emb_np, q_emb_matrix)[0]

# Формируем таблицу для всех вопросов с их сходством по тексту вопроса
df_all = df.copy()
df_all["сходство_вопрос"] = q_sims

print("\nТаблица всех вопросов с косинусным сходством (по тексту вопроса):")
print(df_all[["вопрос", "категория", "сходство_вопрос"]])

# -------------------------------
# 7. Выбираем топ-10 вопросов по сходству текста
# -------------------------------
num_top = min(len(q_sims), 10)
top10_indices = np.argsort(q_sims)[::-1][:num_top]
top10_q_sims = q_sims[top10_indices]

# -------------------------------
# 8. Для топ-10 вопросов вычисляем сходство между запросом и их категориями
# -------------------------------
top10_cat_emb = cat_emb_matrix[top10_indices]
top10_c_sims = cosine_similarity(user_q_emb_np, top10_cat_emb)[0]

# Формируем DataFrame для топ-10
top10_df = pd.DataFrame({
    "index": top10_indices,
    "вопрос": df.iloc[top10_indices]["вопрос"].values,
    "категория": df.iloc[top10_indices]["категория"].values,
    "сходство_вопрос": top10_q_sims,
    "сходство_категория": top10_c_sims
})

# -------------------------------
# 9. Определяем "выбранную" категорию
# Выбираем ту категорию, для которой сходство между запросом и категорией максимально среди топ-10
# -------------------------------
best_cat_rel_idx = np.argmax(top10_c_sims)
best_category = top10_df.loc[best_cat_rel_idx, "категория"]

# Получаем эмбеддинг выбранной категории (по оригинальному индексу вопроса)
best_category_orig_idx = top10_df.loc[best_cat_rel_idx, "index"]
best_category_emb = df.loc[best_category_orig_idx, "category_embedding"].numpy().reshape(1, -1)

# -------------------------------
# 10. Вычисляем косинусное сходство между эмбеддингами категорий топ-10 и выбранной категорией
# -------------------------------
top10_cat_sim_to_best = cosine_similarity(best_category_emb, top10_cat_emb)[0]
top10_df = top10_df.assign(сходство_категории_к_выбранной=top10_cat_sim_to_best)

# -------------------------------
# 11. Сортируем топ-10 по новому показателю (сходство категорий к выбранной)
# -------------------------------
sorted_df = top10_df.sort_values(by="сходство_категории_к_выбранной", ascending=False)

# -------------------------------
# 12. Выбираем финальный список - топ-N вопросов (если найденных меньше, то выводятся все)
# -------------------------------
top_n = 5
final_top = sorted_df.head(top_n)

# -------------------------------
# 13. Вывод результатов
# -------------------------------
print("\nЛог (топ-10 вопросов с метриками):")
print(top10_df)

print("\nВыбранная категория (на основе максимального сходства по категории):", best_category)

print("\nФинальный список вопросов (топ по сходству категорий к выбранной):")
print(final_top)

# -------------------------------
# 14. Вывод таблицы с косинусными расстояниями для всех вопросов
# -------------------------------
# В данной таблице для каждого вопроса указано сходство текста вопроса с запросом пользователя.
# Можно расширить таблицу, добавив и другие метрики по необходимости.
print("\nПолная таблица с косинусными расстояниями для всех вопросов:")
print(df_all[["вопрос", "категория", "сходство_вопрос"]])
