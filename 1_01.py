0 9 * * * /usr/bin/python3 /путь/до/скрипта.py
import time
from datetime import datetime, timedelta

def time_until_end():
    now = datetime.now()
    # Время окончания — сегодня в 17:00
    end_time = now.replace(hour=17, minute=0, second=0, microsecond=0)
    if now > end_time:
        # Если время уже прошло, то возвращаем 0
        return 0
    return (end_time - now).total_seconds()

def main():
    duration = time_until_end()
    print(f"Скрипт будет работать примерно {duration / 60:.2f} минут.")
    
    start_time = time.time()
    while time.time() - start_time < duration:
        # Здесь основная логика скрипта
        print("Работаю...")
        time.sleep(10)  # пауза между итерациями, чтобы не грузить процессор

    print("Время работы завершено — выхожу.")

if __name__ == "__main__":
    main()
