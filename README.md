# Telegram-бот: ИИ-аналитик таблиц и графиков

Бот принимает Excel, CSV/TSV/TXT, фотографии таблиц и текстовые таблицы. Он очищает типичные ошибки форматов, предлагает график по смыслу, строит PNG, экспортирует Excel и формирует локальную статистику. При наличии Groq API ключа распознаёт таблицы на фото и добавляет ИИ-комментарий.

## Возможности
- Вход: `.xlsx`, `.xlsm`, `.csv`, `.tsv`, `.txt`, фотографии и таблицы, вставленные текстом.
- Улучшенная обработка чисел с десятичной запятой/точкой, пустых строк, повторяющихся заголовков и дат.
- Автоподбор: линия, столбцы, доли, распределение, связь числовых показателей.
- Графики: линейный, столбчатый, накопительный, область, scatter с линией тренда, круговой, гистограмма, boxplot, тепловая карта корреляций.
- Статистика: среднее, медиана, min/max, пропуски, повторяющиеся строки, потенциальные выбросы, корреляции.
- Экспорт PNG + Excel с данными и изображением графика.
- Groq API используется для OCR фото и ИИ-комментария. Локальная обработка и графики работают и без API-ключа.

## Настройка Groq
1. Создай ключ на https://console.groq.com/keys.
2. В FadeHost → Environment Variables добавь `GROQ_API_KEY`.
3. Модель по умолчанию `qwen/qwen3.8-27b` — проверь, что она доступна в твоём Groq Console и поддерживает vision в твоём аккаунте. При необходимости поменяй `GROQ_MODEL` на актуальную vision-модель из каталога Groq.
4. Не помещай настоящий ключ в GitHub. Ключ добавляется только в секреты FadeHost.

Groq имеет бесплатный план с ограничениями и может менять доступность моделей/квоты. API не гарантированно бесплатен без ограничений.

## Развёртывание на FadeHost
- Runtime: Python
- Install dependencies: `pip install -r requirements.txt`
- Start command: `python bot.py`
- Environment variables:
  - `TELEGRAM_BOT_TOKEN` — токен Telegram-бота от @BotFather.
  - `GROQ_API_KEY` — API key Groq.
  - `GROQ_MODEL` — необязательно, по умолчанию `qwen/qwen3.8-27b`.

## Обновить существующий GitHub-репозиторий на Mac
Скачай ZIP, распакуй его. Скопируй файлы из папки проекта в существующий клон репозитория `telegram-table-chart-bot`, а затем в Terminal:

```bash
cd ~/Downloads/telegram-table-chart-bot
git status
git add bot.py requirements.txt .gitignore .env.example README.md
git commit -m "Upgrade table analysis and Groq vision"
git push
```

Если ты распаковывала ZIP в `~/Downloads/telegram-table-chart-agent`, скопируй файлы так:

```bash
cp ~/Downloads/telegram-table-chart-agent/bot.py ~/Downloads/telegram-table-chart-bot/
cp ~/Downloads/telegram-table-chart-agent/requirements.txt ~/Downloads/telegram-table-chart-bot/
cp ~/Downloads/telegram-table-chart-agent/.gitignore ~/Downloads/telegram-table-chart-bot/
cp ~/Downloads/telegram-table-chart-agent/.env.example ~/Downloads/telegram-table-chart-bot/
cp ~/Downloads/telegram-table-chart-agent/README.md ~/Downloads/telegram-table-chart-bot/
```

Не запускай повторно `git init` или `git remote add origin`, если репозиторий уже настроен.
