def clean_and_group_dialogs(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    dialogs, current_dialog = [], []
    current_user, last_message, last_speaker = None, None, None

    exclude_phrases = [
        "№:", "Начат:", "Отделы:", "Посетитель переведён в статус офлайн.",
        "Диалогу присвоена категория", "Приветствуем Вас в чате Просрочки.net!",
        "Здесь консультируют по вопросам, связанным с просроченным долгом",
        "Режим работы: пн-чт 9:00-17.00; пт 9.00-16.00.", "Будем рады помочь!",
        "Диалог автоназначен на агента", "Диалог завершен", "https://viber.com/",
        "Мы не получали от вас сообщения более 15 минут.",
        "-----------", 'Имя:', "Агент Альфа-Банк (просрочка) включился в разговор"
    ]
    speaker_pattern = re.compile(r'^\d{2}:\d{2}:\d{2} ([^:]+): (.+)')

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if any(phrase in line for phrase in exclude_phrases):
            continue

        match_user = re.search(r'Имя посетителя: (.+)', line)
        if match_user:
            user = match_user.group(1)

            if current_user is None:
                current_user = user
            elif current_user != user:
                if last_message:
                    current_dialog.append(last_message)
                if current_dialog:
                    dialogs.append((current_user, current_dialog))
                current_dialog = []
                current_user = user
                last_message = None
                last_speaker = None
            continue

        match_speaker = speaker_pattern.match(line)
        if match_speaker:
            speaker = match_speaker.group(1)
            text = match_speaker.group(2)

            if last_message and (speaker == last_speaker or ("Альфа" in speaker and "Альфа" in last_speaker)):
                last_message += " " + text
            else:
                if last_message:
                    current_dialog.append(last_message)
                last_message = f"{speaker}: {text}"
                last_speaker = speaker
        else:
            if last_message is None:
                last_message = line
            else:
                last_message += " " + line

    if last_message:
        current_dialog.append(last_message)
    if current_dialog:
        dialogs.append((current_user, current_dialog))

    return dialogs