import pandas as pd
from sentence_transformers import SentenceTransformer
from pymorphy2 import MorphAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

morph = MorphAnalyzer()
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  # Работает с русским

def extract_morph_tags(text):
    tags = []
    for word in text.split():
        parsed = morph.parse(word)[0]
        tags.append(str(parsed.tag))
    return tags

def morph_diff_penalty(tags1, tags2):
    # Чем больше различий — тем больше штраф
    diff_count = sum(1 for t1, t2 in zip(tags1, tags2) if t1 != t2)
    max_len = max(len(tags1), len(tags2))
    return diff_count / max_len if max_len else 0

def find_best_answer(df: pd.DataFrame, user_question: str):
    # Векторизация вопросов
    question_embeddings = model.encode(df["вопрос"].tolist())
    user_embedding = model.encode([user_question])[0]

    # Морфологический анализ
    user_tags = extract_morph_tags(user_question)
    penalties = []

    for q in df["вопрос"]:
        q_tags = extract_morph_tags(q)
        penalty = morph_diff_penalty(user_tags, q_tags)
        penalties.append(penalty)

    # Косинусная близость
    similarities = cosine_similarity([user_embedding], question_embeddings)[0]

    # Итоговая метрика — с учётом штрафа
    adjusted_scores = similarities - np.array(penalties)

    # Выбор наилучшего ответа
    best_idx = np.argmax(adjusted_scores)
    best_row = df.iloc[best_idx]

    return {
        "похожий_вопрос": best_row["вопрос"],
        "ответ": best_row["ответ"],
        "схожесть": similarities[best_idx],
        "штраф": penalties[best_idx],
        "итоговый_балл": adjusted_scores[best_idx]
    }

