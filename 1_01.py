morph = pymorphy3.MorphAnalyzer()

WEEKDAYS_LEMMAS = {'понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье'}
MONTH_PATTERN = r'\d{1,2}\s*(январ[яею]|феврал[яею]|март[аеу]?|апрел[яею]|ма[йяею]|июн[яею]|июл[яею]|август[аеу]?|сентябр[яею]|октябр[яею]|ноябр[яею]|декабр[яею])'
DIGIT_DATE_PATTERN = r'\b\d{1,2}\.\d{1,2}(?:\.\d{2,4})?\b'
TIME_POINTERS = (
    r'(этой|этот|этом|этому|этого|следующей|следующий|следующем|следующему|следующего|'
    r'прошлой|прошлый|прошлом|прошлому|прошлого|той|тот|том|тому|того|будущей|будущий|будущем|будущему|будущего)'
)
TIME_UNITS = r'(недел[аеиюе]|месяц[аеуыио]?|квартал[аеуыио]?)'
DATE_PREPOSITIONS = r'(до|к|в|по|на|около|примерно|после)?\s*'

RELATIVE_DATE_PATTERN = rf'{DATE_PREPOSITIONS}{TIME_POINTERS}\s+{TIME_UNITS}'

def replace_dates(text):
    text = re.sub(DIGIT_DATE_PATTERN, '[DATE]', text)
    text = re.sub(rf'{DATE_PREPOSITIONS}?{MONTH_PATTERN}', '[DATE]', text, flags=re.IGNORECASE)
    text = re.sub(RELATIVE_DATE_PATTERN, '[DATE]', text, flags=re.IGNORECASE)

    words = text.split()
    for i, word in enumerate(words):
        parsed = morph.parse(word.strip('.,!?:;()'))[0]
        if parsed.normal_form in WEEKDAYS_LEMMAS:
            if i > 0 and words[i-1].lower() in ['до', 'в', 'по', 'на', 'к']:
                words[i-1] = '[DATE]'
                words[i] = ''
            else:
                words[i] = '[DATE]'
    text = ' '.join(w for w in words if w)
    return text


import re

def remove_emojis(text):
    # Универсальный диапазон символов для большинства эмодзи
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Смайлы
        "\U0001F300-\U0001F5FF"  # Символы и пиктограммы
        "\U0001F680-\U0001F6FF"  # Транспорт
        "\U0001F1E0-\U0001F1FF"  # Флаги
        "\U00002700-\U000027BF"  # Разные символы
        "\U0001F900-\U0001F9FF"  # Эмоции, жесты и т.п.
        "\U0001FA70-\U0001FAFF"  # Доп. символы
        "\U00002500-\U00002BEF"  # Разные иероглифы
        "\U0000200D"             # Zero Width Joiner
        "\U00002300-\U000023FF"  # Технические символы
        "\U0000FE00-\U0000FE0F"  # Вариации эмодзи
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def replace_phones(text):
    PATTERN_COMPACT = r'(?<!\d)(?:\+375(25|29|33|44)\d{7}|375(25|29|33|44)\d{7}|80(25|29|33|44)\d{7})(?!\d)'
    PATTERN_FLEX = r'(?<!\d)(?:(?:\+?375|375)[\s\-()]*\(?(25|29|33|44)\)?(?:[\s\-()]?\d){7}|8[\s\-()]*0?\(?(25|29|33|44)\)?(?:[\s\-()]?\d){7})(?!\d)'

    text = re.sub(PATTERN_COMPACT, '[TEL]', text, flags=re.VERBOSE)
    text = re.sub(PATTERN_FLEX, '[TEL]', text, flags=re.VERBOSE)
    return text
