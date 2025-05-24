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

def replace_dates(text):
    text = re.sub(DIGIT_DATE_PATTERN, '[DATE]', text)
    text = re.sub(rf'{DATE_PREPOSITIONS}?{MONTH_PATTERN}', '[DATE]', text, flags=re.IGNORECASE)
    text = re.sub(RELATIVE_DATE_PATTERN, '[DATE]', text, flags=re.IGNORECASE)
    text = re.sub(NUMERIC_DAY_PATTERN, '[DATE]', text, flags=re.IGNORECASE)

    words = text.split()
    for i, word in enumerate(words):
        clean_word = word.strip('.,!?:;()').lower()
        parsed = morph.parse(clean_word)[0]

        # Замена дней недели с предлогом
        if parsed.normal_form in WEEKDAYS_LEMMAS:
            if i > 0 and words[i-1].lower() in ['до', 'в', 'по', 'на', 'к']:
                words[i-1] = '[DATE]'
                words[i] = ''
            else:
                words[i] = '[DATE]'
        # Замена слов сегодня, завтра, послезавтра
        elif clean_word in RELATIVE_WORDS:
            words[i] = '[DATE]'
    text = ' '.join(w for w in words if w)
    return text

def replace_dates_with_iteration(text, replace_dates_func):
    first_pass = replace_dates_func(text)
    temp_text = first_pass.replace('[DATE]', ' ')
    second_pass = replace_dates_func(temp_text)
    if second_pass.count('[DATE]') > 0:
        return second_pass
    return first_pass

def replace_phones(text):
    text = re.sub(PATTERN_COMPACT, '[TEL]', text, flags=re.VERBOSE)
    text = re.sub(PATTERN_FLEX, '[TEL]', text, flags=re.VERBOSE)
    return text

def replace_names(text):
    for pattern in [PATTERN_FULL, PATTERN_INITIALS, PATTERN_NAME_PATRONYM, PATTERN_PATRONYM_ONLY]:
        text = re.sub(pattern, '[NAME]', text)
    return re.sub(pattern, '[NAME]', text)

def remove_links(text):
    return re.sub(r'https?://\S+', '', text)
