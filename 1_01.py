morph = pymorphy3.MorphAnalyzer()

WEEKDAYS_LEMMAS = {'понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье'}
RELATIVE_WORDS = {'сегодня', 'завтра', 'послезавтра'}
MONTH_PATTERN = (r'\d{1,2}\s*(январ[яею]|феврал[яею]|март[аеу]?|апрел[яею]|ма[йяею]|июн[яею]|июл[яею]|август[аеу]?'
                 r'|сентябр[яею]|октябр[яею]|ноябр[яею]|декабр[яею])')
DIGIT_DATE_PATTERN = r'\b\d{1,2}\.\d{1,2}(?:\.\d{2,4})?\b'
TIME_POINTERS = (
    r'(этой|этот|этом|этому|этого|следующей|следующий|следующем|следующему|следующего|'
    r'прошлой|прошлый|прошлом|прошлому|прошлого|той|тот|том|тому|того|будущей|будущий|будущем|будущему|будущего)'
)
TIME_UNITS = r'(недел[аеиюе]|месяц[аеуыио]?|квартал[аеуыио]?)'
DATE_PREPOSITIONS = r'(до|к|в|по|на|около|примерно|после)?\s*'
NUMERIC_DAY_PATTERN = r'\b(?:до|в|на|по|к)?-?\s*\d{1,2}[\s\-]?(?:го|числ[аоуе]?)\b'
RELATIVE_DATE_PATTERN = rf'{DATE_PREPOSITIONS}{TIME_POINTERS}\s+{TIME_UNITS}'

PATTERN_COMPACT = r'(?<!\d)(?:\+375(25|29|33|44)\d{7}|375(25|29|33|44)\d{7}|80(25|29|33|44)\d{7}|\+7\d{10}|7\d{10}|8\d{10})(?!\d)'
PATTERN_FLEX = (
    r'(?<!\d)('
    r'(?:\+?375|375)[\s\-()]*\(?(25|29|33|44)\)?(?:[\s\-()]?\d){7}'  # Беларусь
    r'|8[\s\-()]*0?\(?(25|29|33|44)\)?(?:[\s\-()]?\d){7}'            # Беларусь через 8
    r'|\+?7[\s\-()]*(?:\d{3})[\s\-()]*(?:\d{3})[\s\-()]*(?:\d{2})[\s\-()]*(?:\d{2})'  # Россия +7
    r'|8[\s\-()]*(?:\d{3})[\s\-()]*(?:\d{3})[\s\-()]*(?:\d{2})[\s\-()]*(?:\d{2})'     # Россия 8
    r')(?!\d)'
)

PATTERN_FULL = r'\b[А-ЯЁ][а-яё]+[\s\-]+[А-ЯЁ][а-яё]+[\s\-]+[А-ЯЁ][а-яё]+\b'
PATTERN_INITIALS = r'\b[А-ЯЁ][а-яё]+[\s\-]+[А-ЯЁ]\.?(?:\s?[А-ЯЁ]\.?){0,1}\b'
PATTERN_NAME_PATRONYM = r'\b[А-ЯЁ][а-яё]+[\s\-]+[А-ЯЁ][а-яё]+(вич(ем|а|у|е|и)?|вн(а|ой|е|у|и)?)\b'
PATTERN_PATRONYM_ONLY = r'\b[А-ЯЁ][а-яё]+(вич(ем|а|у|е|и)?|вн(а|ой|е|у|и)?)\b'

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
        "]+", flags=re.UNICODE)

GREETING_PATTERN = r'\bпривет\b|\bздра?вствуй(те)?\b|\bдобрый (день|вечер|утро)\b|\bздравствуйте\b|\bздарова\b|\bхай\b|\bалло\b|\bдарова\b|\bприв\b|\bсалют\b|\bдоброе утро\b|\bдобрый вечер\b|\bдобрый день\b'
GREETING_REGEX = re.compile('|'.join(GREETING_PATTERNS), re.IGNORECASE)

def replace_dates(text):
    """Заменяет различные формы дат (числовые, словесные, относительные) на [DATE]."""
    text = re.sub(DIGIT_DATE_PATTERN, '[DATE]', text)
    text = re.sub(rf'{DATE_PREPOSITIONS}?{MONTH_PATTERN}', '[DATE]', text, flags=re.IGNORECASE)
    text = re.sub(RELATIVE_DATE_PATTERN, '[DATE]', text, flags=re.IGNORECASE)
    text = re.sub(NUMERIC_DAY_PATTERN, '[DATE]', text, flags=re.IGNORECASE)

    words = text.split()
    for i, word in enumerate(words):
        clean_word = word.strip('.,!?:;()').lower()
        parsed = morph.parse(clean_word)[0]

        if parsed.normal_form in WEEKDAYS_LEMMAS:
            if i > 0 and words[i-1].lower() in ['до', 'в', 'по', 'на', 'к']:
                words[i-1] = '[DATE]'
                words[i] = ''
            else:
                words[i] = '[DATE]'
        elif clean_word in RELATIVE_WORDS:
            words[i] = '[DATE]'
    text = ' '.join(w for w in words if w)
    return text


def replace_dates_with_iteration(text, replace_dates_func):
    """Выполняет замену дат с повторной проверкой, если [DATE] уже была найдена."""
    first_pass = replace_dates_func(text)
    temp_text = first_pass.replace('[DATE]', ' ')
    second_pass = replace_dates_func(temp_text)
    if second_pass.count('[DATE]') > 0:
        return second_pass
    return first_pass


def replace_phones(text):
    """Заменяет номера телефонов (BY/RU) на [TEL]."""
    text = re.sub(PATTERN_COMPACT, '[TEL]', text, flags=re.VERBOSE)
    text = re.sub(PATTERN_FLEX, '[TEL]', text, flags=re.VERBOSE)
    return text


def replace_names(text):
    """Заменяет ФИО и отчества на [NAME]."""
    for pattern in [PATTERN_FULL, PATTERN_INITIALS, PATTERN_NAME_PATRONYM, PATTERN_PATRONYM_ONLY]:
        text = re.sub(pattern, '[NAME]', text)
    return re.sub(pattern, '[NAME]', text)


def remove_links(text):
    """Удаляет ссылки http/https из текста."""
    return re.sub(r'https?://\S+', '', text)


def remove_emojis(text):
    """Удаляет все emoji из текста."""
    return emoji_pattern.sub('', text)


def replace_amounts(text):
    """Заменяет суммы в рублях, долларах, юанях и т.п. на [AMOUNT]."""

    amount_keywords = r'(оплат[аилюе]?|плат[аилюе]?|перевел[аи]?|перевод|заплат[аилюе]?|сумма|стоимость|цен[аы]|буду платить|долг)'
    currencies = r'(р(уб\.?|убля[хм]?)?|бел\.?\s?р(уб\.?)?|белорусских?\s?рубл(ей|я)?|российских?\s?рубл(ей|я)?|₽|BYN|RUB|USD|\$|доллар[аовы]*|CNY|юан[ейя])'
    amount_pattern = rf'\b\d+[.,]?\d*\s*{currencies}?\b'

    def mark_amounts_around_keywords(text):
        pattern = rf'({amount_keywords})[\s:,-]*({amount_pattern})|({amount_pattern})[\s:,-]*({amount_keywords})'
        return re.sub(pattern, '[AMOUNT]', text, flags=re.IGNORECASE)

    def mark_standalone_amounts(text):
        return re.sub(amount_pattern, '[AMOUNT]', text, flags=re.IGNORECASE)

    text = mark_amounts_around_keywords(text)
    text = mark_standalone_amounts(text)
    return text


def remove_greetings(text):
    """Удаляет из текста распространённые приветствия."""
    return GREETING_REGEX.sub('', text).strip()


def remove_punctuation_except_last(text):
    """Удаляет все знаки препинания из текста, кроме последнего.
    Если в конце текста нет знака препинания, оставляет текст без изменений. """

    match = re.search(r'([.,!?;:])\s*$', text)
    last_punct = match.group(1) if match else ''
    text_wo_punct = re.sub(r'[.,!?;:]', '', text)
    text = text_wo_punct.rstrip() + last_punct if last_punct else text_wo_punct
    return text.strip()
