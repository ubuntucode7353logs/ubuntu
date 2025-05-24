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
