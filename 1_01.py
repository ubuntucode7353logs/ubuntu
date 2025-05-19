import re
import pandas as pd

# -------------------------
# Глобальные паттерны
# -------------------------

FIO_PATTERN = r'((?:[А-ЯЁ][а-яё]+(?:\s?))+){2,3}'

GREETINGS = [
    'привет', 'приет', 'прив', 'здравствуйте', 'здраствуйте', 'драсте',
    'добрый день', 'добрый вечер', 'доброе утро', 'доброго дня', 'день добрый',
    'здорово', 'хай', 'ку', 'хелло', 'салам', 'здарова', 'даров', 'здоров'
]
GREETINGS_PATTERN = r'\b(?:' + '|'.join(re.escape(g) for g in GREETINGS) + r')\b[\s,.:;!?-]*'

DATE_PATTERNS = [
    r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b',
    r'\b\d{4}-\d{2}-\d{2}\b'
]

MONEY_PATTERN = r'\b\d+[.,]?\d*\s*(?:₽|rub\.?|руб\.?|рублей|рубля|руб)?[.,]?\b'

ACCOUNT_PATTERN = r'\b3\d{16}\b'

PHONE_PATTERN = r'(?:\+375|80)(?:\s?|\-?)\d{2}(?:\s?|\-?)\d{3}(?:\s?|\-?)\d{2}(?:\s?|\-?)\d{2}'

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & pictographs
    "\U0001F680-\U0001F6FF"  # Transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

# -------------------------
# Функции очистки
# -------------------------

def remove_names(text):
    return re.sub(FIO_PATTERN, '', text).strip()

def remove_greetings(text):
    text = re.sub(GREETINGS_PATTERN, '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

def replace_dates(text):
    for pattern in DATE_PATTERNS:
        text = re.sub(pattern, '<DATE>', text)
    return text

def replace_money(text):
    return re.sub(MONEY_PATTERN, '<MONEY>', text, flags=re.IGNORECASE)

def remove_account_numbers(text):
    return re.sub(ACCOUNT_PATTERN, '<ACCOUNT>', text)

def remove_phone_numbers(text):
    return re.sub(PHONE_PATTERN, '<PHONE>', text)

def remove_emojis(text):
    return EMOJI_PATTERN.sub('', text)

def remove_punctuation(text):
    return re.sub(r'[^\w\s<>]', '', text)

# -------------------------
# Основная очистка текста
# -------------------------

def preprocess_text(text):
    text = remove_names(text)
    text = remove_greetings(text)
    text = replace_dates(text)
    text = replace_money(text)
    text = remove_account_numbers(text)
    text = remove_phone_numbers(text)
    text = remove_emojis(text)
    text = remove_punctuation(text)
    return text.strip()

# -------------------------
# Обработка диалогов из файла
# -------------------------

def parse_dialogs_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    blocks = re.split(r'\n\s*\n', text.strip())
    records = []

    for block in blocks:
        try:
            index_match = re.search(r'№:\s*(\d+)', block)
            index = index_match.group(1)
            name_match = re.search(r'Имя:\s*(.+)', block)
            user_name = name_match.group(1).strip()
            pattern = rf'\d{{2}}:\d{{2}}:\d{{2}} {re.escape(user_name)}:\s(.+)'
            user_messages = re.findall(pattern, block)
            cleaned = preprocess_text(" ".join(user_messages))
            print(cleaned)
            if len(cleaned) > 0 and (has_personal_verbs(cleaned) or is_personal_request(cleaned)):
                records.append({'index': index, 'message': cleaned})
        except (AttributeError, TypeError):
            continue

    df = pd.DataFrame(records)
    df['message'].dropna(inplace=True)
    return df
