import torch
from transformers import AutoTokenizer, AutoModel

# Mean pooling (учитывает attention_mask)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # эмбеддинги токенов
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
           torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

# Функция: получить эмбеддинг вопроса
def get_question_embedding(line, tokenizer, model):
    encoded_input = tokenizer(line, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embedding

# Функция: косинусное сходство
def cosine_similarity(tensor_a, tensor_b):
    dot_product = torch.dot(tensor_a.squeeze(), tensor_b.squeeze())
    norm_a = torch.norm(tensor_a)
    norm_b = torch.norm(tensor_b)
    cos_sim = dot_product / (norm_a * norm_b + 1e-9)
    return cos_sim

# Функция: получить топ-5 наиболее похожих вопросов
def get_top_5_questions(line, question_list_tensor, tokenizer, model):
    cosine_similarity_all = []
    question_tensor = get_question_embedding(line, tokenizer, model)

    for i in range(len(question_list_tensor)):
        sim = cosine_similarity(question_tensor, question_list_tensor[i])
        cosine_similarity_all.append(sim)

    cosine_similarity_all = torch.stack(cosine_similarity_all)
    top_values, top_indices = torch.topk(cosine_similarity_all, 5, largest=True)

    return top_values, top_indices


