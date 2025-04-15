import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pymorphy2

morph = pymorphy2.MorphAnalyzer()
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Хорошо работает для ru/en

# Простые стоп-слова, можно расширить
STOP_WORDS = set([
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со",
    "как", "а", "то", "все", "она", "так", "его", "но", "да", "ты"
])


def clean_and_lemmatize(text):
    # Убираем лишнее и лемматизируем
    tokens = re.findall(r'\b\w+\b', text.lower())
    lemmas = [morph.parse(token)[0].normal_form for token in tokens if token not in STOP_WORDS]
    return lemmas


def compute_coverage(base_q_lemmas, client_q_lemmas):
    if not base_q_lemmas:
        return 0
    overlap = set(base_q_lemmas) & set(client_q_lemmas)
    return len(overlap) / len(base_q_lemmas)


def match_question(client_question, knowledge_base, coverage_threshold=0.75):
    client_lemmas = clean_and_lemmatize(client_question)
    candidates = []

    for base_q, answer in knowledge_base.items():
        base_lemmas = clean_and_lemmatize(base_q)
        coverage = compute_coverage(base_lemmas, client_lemmas)
        if coverage >= coverage_threshold:
            candidates.append((base_q, answer, coverage))

    if not candidates:
        return None  # или сообщение об уточнении

    # Доработка: ранжируем по косинусной близости
    client_embed = model.encode([client_question])
    candidate_questions = [q for q, _, _ in candidates]
    candidate_embeds = model.encode(candidate_questions)
    sims = cosine_similarity(client_embed, candidate_embeds)[0]

    best_idx = np.argmax(sims)
    return {
        'вопрос_из_базы': candidates[best_idx][0],
        'ответ': candidates[best_idx][1],
        'coverage': candidates[best_idx][2],
        'similarity': sims[best_idx]
    }

def clean_and_lemmatize(text):
    # Токенизация и лемматизация без удаления стоп-слов
    tokens = re.findall(r'\b\w+\b', text.lower())
    lemmas = [morph.parse(token)[0].normal_form for token in tokens]
    return lemmas
