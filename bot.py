import asyncio, io, json, logging, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from google import genai
from PIL import Image
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("table-chart-bot")

NAMES = {"line":"Линейный график", "bar":"Столбчатая диаграмма", "scatter":"Точечная диаграмма", "pie":"Круговая диаграмма", "hist":"Гистограмма"}
KEYBOARD = InlineKeyboardMarkup([
    [InlineKeyboardButton("📈 Линейный", callback_data="chart:line"), InlineKeyboardButton("📊 Столбчатый", callback_data="chart:bar")],
    [InlineKeyboardButton("🔵 Точечный", callback_data="chart:scatter"), InlineKeyboardButton("🥧 Круговой", callback_data="chart:pie")],
    [InlineKeyboardButton("📉 Гистограмма", callback_data="chart:hist")]
])

def clean_df(df):
    df = df.copy().dropna(how="all").dropna(axis=1, how="all")
    if df.empty or len(df.columns) == 0: raise ValueError("В таблице не найдены данные.")
    df.columns = [str(c).strip() or f"Столбец {i+1}" for i, c in enumerate(df.columns)]
    for col in df.columns:
        if df[col].dtype == object:
            orig = df[col]
            norm = orig.astype(str).str.strip().str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
            vals = pd.to_numeric(norm, errors="coerce")
            nonempty = orig.notna() & orig.astype(str).str.strip().ne("")
            if nonempty.sum() and vals[nonempty].notna().sum() / nonempty.sum() >= .8: df[col] = vals
    return df.head(5000).iloc[:, :30].reset_index(drop=True)

def parse_text(text):
    text = text.strip()
    if not text: raise ValueError("Пустое сообщение.")
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if len(lines) >= 2 and all("|" in x for x in lines[:2]):
        text = "\n".join(x.strip().strip("|") for x in lines)
        df = pd.read_csv(io.StringIO(text), sep=r"\s*\|\s*", engine="python")
    else:
        try: df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except Exception: df = pd.read_csv(io.StringIO(text), sep=";", engine="python")
    return clean_df(df)

def parse_file(filename, content):
    low = filename.lower()
    if low.endswith((".xlsx", ".xlsm")): df = pd.read_excel(io.BytesIO(content))
    elif low.endswith((".csv", ".txt")):
        text = None
        for enc in ("utf-8-sig", "cp1251", "latin-1"):
            try: text = content.decode(enc); break
            except UnicodeDecodeError: pass
        if text is None: raise ValueError("Не удалось прочитать файл. Сохраните CSV в UTF-8.")
        return parse_text(text)
    else: raise ValueError("Поддерживаются Excel (.xlsx, .xlsm) и CSV (.csv, .txt).")
    return clean_df(df)

def ocr_photo(content):
    if not GEMINI_KEY: raise RuntimeError("Для фото таблиц добавь GEMINI_API_KEY в настройках FadeHost.")
    client = genai.Client(api_key=GEMINI_KEY)
    image = Image.open(io.BytesIO(content)).convert("RGB")
    prompt = '''Перенеси видимую таблицу с изображения в JSON. Верни ТОЛЬКО JSON формата {"columns":["Имя","Значение"],"rows":[["A",12],["B",15]]}. Не придумывай значения. Пустые/нечитаемые ячейки — null. Сохраняй порядок. Если таблицу невозможно прочитать, верни {"columns":[],"rows":[]}'''
    response = client.models.generate_content(model=GEMINI_MODEL, contents=[prompt, image])
    raw = (response.text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    if not data.get("columns") or not data.get("rows"): raise ValueError("Не удалось распознать таблицу. Попробуй сделать фото чётче.")
    return clean_df(pd.DataFrame(data["rows"], columns=data["columns"]))

def nums(df):
    return [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def make_chart(df, kind):
    numeric = nums(df); first = str(df.columns[0])
    if kind == "scatter":
        if len(numeric) < 2: raise ValueError("Для точечной диаграммы нужны два числовых столбца.")
        x, y = numeric[:2]
    elif kind == "hist":
        if not numeric: raise ValueError("Для гистограммы нужен числовой столбец.")
        x, y = numeric[0], None
    elif kind == "pie":
        if not numeric: raise ValueError("Для круговой диаграммы нужен числовой столбец.")
        x, y = (first if first != numeric[0] else None), numeric[0]
    else:
        if not numeric: raise ValueError("Не найден числовой столбец.")
        x, y = first, next((c for c in numeric if c != first), numeric[0])
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    if kind == "hist":
        v = pd.to_numeric(df[x], errors="coerce").dropna()
        if v.empty: raise ValueError("Нет числовых значений.")
        ax.hist(v, bins=min(20, max(5, int(len(v)**.5))), edgecolor="black")
        ax.set_xlabel(x); ax.set_ylabel("Количество"); ax.set_title(f"Распределение: {x}")
    elif kind == "scatter":
        d = df[[x,y]].apply(pd.to_numeric, errors="coerce").dropna()
        if d.empty: raise ValueError("Нет пар числовых значений.")
        ax.scatter(d[x], d[y], alpha=.8); ax.set_xlabel(x); ax.set_ylabel(y)
        ax.set_title(f"{y} в зависимости от {x}"); ax.grid(True, alpha=.25)
    elif kind == "pie":
        labels = df[x].fillna("Без названия").astype(str).tolist() if x else [str(i+1) for i in range(len(df))]
        values = pd.to_numeric(df[y], errors="coerce").fillna(0)
        valid = values > 0
        labels = [label for label, ok in zip(labels, valid) if ok]; values = values[valid]
        if values.empty: raise ValueError("Для круговой диаграммы нужны положительные значения.")
        if len(values) > 12:
            labels = labels[:11] + ["Прочее"]
            values = pd.concat([values.iloc[:11], pd.Series([values.iloc[11:].sum()])], ignore_index=True)
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90); ax.set_title(f"Доли: {y}")
    else:
        d = df[[x,y]].dropna().copy()
        d[y] = pd.to_numeric(d[y], errors="coerce"); d = d.dropna(subset=[y]).head(100)
        if d.empty: raise ValueError("Нет строк с числовыми значениями.")
        xs = d[x].astype(str)
        if kind == "line": ax.plot(xs, d[y], marker="o", linewidth=2)
        else: ax.bar(xs, d[y])
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{y} по {x}")
        if len(xs) > 10: ax.tick_params(axis="x", labelrotation=45)
        ax.grid(axis="y", alpha=.25)
    out = io.BytesIO(); fig.savefig(out, format="png", dpi=160, bbox_inches="tight"); plt.close(fig)
    return out.getvalue()

def analysis(df):
    lines = ["📋 Краткий анализ данных", f"• Строк: {len(df)}", f"• Столбцов: {len(df.columns)}", "• Столбцы: " + ", ".join(map(str, df.columns[:12]))]
    for col in nums(df)[:4]:
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        if not v.empty: lines.append(f"• {col}: среднее {v.mean():.2f}; минимум {v.min():.2f}; максимум {v.max():.2f}")
    if not nums(df): lines.append("• Числовые столбцы не обнаружены.")
    if len(df) >= 5000: lines.append("• Обработаны первые 5000 строк.")
    return "\n".join(lines)

def excel_with_chart(df, png):
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Font
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Данные", index=False)
        ws = writer.sheets["Данные"]; ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
        for cells in ws.columns:
            width = max((len(str(c.value)) if c.value is not None else 0) for c in cells)
            ws.column_dimensions[cells[0].column_letter].width = min(max(width+2, 12), 32)
        chart = writer.book.create_sheet("График")
        chart["A1"] = "График создан Telegram-ботом"; chart["A1"].font = Font(bold=True, size=14)
        img = XLImage(io.BytesIO(png)); img.width = 800; img.height = 480; chart.add_image(img, "A3")
    out.seek(0); return out.getvalue()

def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("df", None)
    await update.effective_message.reply_text("Привет! Отправь Excel (.xlsx), CSV, фото таблицы или вставь данные текстом. Затем выбери тип графика — пришлю PNG, Excel с графиком и краткий анализ.\n\n/help — инструкция\n/cancel — сбросить текущую таблицу")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("1) Отправь .xlsx, .csv, фото или таблицу текстом.\n2) Выбери тип графика кнопкой.\n3) Получи изображение, Excel и краткий анализ.\n\nТипы: линейный, столбчатый, точечный, круговой, гистограмма. Для распознавания фото нужен GEMINI_API_KEY.")

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("df", None)
    await update.effective_message.reply_text("Таблица сброшена. Отправь новые данные.")

async def accept_df(update, context, df):
    context.user_data["df"] = df
    preview = df.head(5).to_string(index=False, max_cols=8)
    if len(preview) > 1500: preview = preview[:1500] + "…"
    await update.effective_message.reply_text(f"✅ Таблица прочитана: {len(df)} строк, {len(df.columns)} столбцов.\n\nПервые строки:\n<pre>{esc(preview)}</pre>\n\nВыбери тип графика:", parse_mode="HTML", reply_markup=KEYBOARD)

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message; doc = msg.document; name = doc.file_name or "table.csv"
    if doc.file_size and doc.file_size > 15*1024*1024:
        await msg.reply_text("Файл слишком большой. Максимум 15 МБ."); return
    try:
        f = await doc.get_file(); content = bytes(await f.download_as_bytearray())
        df = await asyncio.to_thread(parse_file, name, content)
        await accept_df(update, context, df)
    except Exception as e:
        log.exception("Ошибка чтения файла"); await msg.reply_text(f"Не удалось прочитать файл: {str(e)[:400]}")

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not GEMINI_KEY:
        await msg.reply_text("Для фото добавь GEMINI_API_KEY в переменные окружения FadeHost. Excel, CSV и текст работают без Gemini."); return
    try:
        f = await msg.photo[-1].get_file(); content = bytes(await f.download_as_bytearray())
        await msg.reply_text("🔎 Распознаю таблицу на фото…")
        df = await asyncio.to_thread(ocr_photo, content)
        await accept_df(update, context, df)
    except Exception as e:
        log.exception("Ошибка распознавания фото")
        await msg.reply_text(f"Не удалось распознать таблицу. Попробуй сделать фото сверху, при хорошем освещении.\nПодробность: {str(e)[:300]}")

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    try:
        df = await asyncio.to_thread(parse_text, msg.text or "")
        if len(df.columns) < 2:
            await msg.reply_text("Нашёл только один столбец. Пришли таблицу с заголовками и минимум двумя столбцами, разделёнными табуляцией, запятой или точкой с запятой."); return
        await accept_df(update, context, df)
    except Exception:
        await msg.reply_text("Не получилось распознать текст как таблицу. Пример:\n<pre>Месяц;Продажи\nЯнварь;120\nФевраль;150\nМарт;135</pre>", parse_mode="HTML")

async def chart_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    kind = q.data.split(":", 1)[1]; df = context.user_data.get("df")
    if df is None:
        await q.message.reply_text("Сначала отправь таблицу."); return
    try:
        png = await asyncio.to_thread(make_chart, df, kind)
        xlsx = await asyncio.to_thread(excel_with_chart, df, png)
        await q.message.reply_photo(photo=io.BytesIO(png), filename="grafik.png", caption=f"✅ {NAMES[kind]}. Строк: {len(df)}, столбцов: {len(df.columns)}.")
        await q.message.reply_document(document=io.BytesIO(xlsx), filename="таблица_и_график.xlsx", caption="Excel: данные и график находятся на отдельных листах.")
        await q.message.reply_text(analysis(df))
        await q.message.reply_text("Можно построить другой тип графика или отправить новую таблицу.", reply_markup=KEYBOARD)
    except Exception as e:
        log.exception("Ошибка построения графика"); await q.message.reply_text(f"Не удалось построить график: {str(e)[:400]}")

def main():
    if not TOKEN: raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN в переменных окружения.")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(CallbackQueryHandler(chart_handler, pattern=r"^chart:(line|bar|scatter|pie|hist)$"))
    app.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    log.info("Бот для построения графиков запущен.")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
