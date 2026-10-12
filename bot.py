import asyncio
import base64
import io
import json
import logging
import os
import re
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from PIL import Image, ImageOps, ImageEnhance
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
MAX_ROWS = 5000
MAX_COLS = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("table-chart-bot")

NAMES = {
    "line": "Линейный график", "bar": "Столбчатая диаграмма", "stacked": "Накопительная диаграмма",
    "area": "График с областями", "scatter": "Точечная диаграмма", "pie": "Круговая диаграмма",
    "hist": "Гистограмма распределения", "box": "Диаграмма размаха", "heatmap": "Тепловая карта корреляций",
}
CHART_ROWS = [
    [InlineKeyboardButton("✨ Подобрать по смыслу", callback_data="chart:auto")],
    [InlineKeyboardButton("📈 Линия", callback_data="chart:line"), InlineKeyboardButton("📊 Столбцы", callback_data="chart:bar")],
    [InlineKeyboardButton("🧱 Накопительные", callback_data="chart:stacked"), InlineKeyboardButton("🌊 Области", callback_data="chart:area")],
    [InlineKeyboardButton("🔵 Связь X–Y", callback_data="chart:scatter"), InlineKeyboardButton("🥧 Доли", callback_data="chart:pie")],
    [InlineKeyboardButton("📉 Распределение", callback_data="chart:hist"), InlineKeyboardButton("📦 Размах", callback_data="chart:box")],
    [InlineKeyboardButton("🌡 Корреляции", callback_data="chart:heatmap")],
]
KEYBOARD = InlineKeyboardMarkup(CHART_ROWS)


def unique_columns(columns):
    result, seen = [], {}
    for i, value in enumerate(columns):
        name = str(value).strip() or f"Столбец {i + 1}"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        result.append(name)
    return result


def coerce_number_series(series):
    """Parse common US/EU numeric formats without changing genuinely textual columns."""
    orig = series.astype("string").str.strip().str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
    def norm_one(v):
        if pd.isna(v) or v in ("", "nan", "None", "<NA>"):
            return None
        s = str(v).replace("%", "")
        # European thousands + decimal: 1.234,56 -> 1234.56
        if "," in s and "." in s and s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        # A single comma with 1-2 trailing digits is usually a decimal comma.
        elif "," in s and "." not in s:
            if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", s):
                s = s.replace(",", "")
            else:
                s = s.replace(",", ".")
        # US thousands separators: 1,234.56
        elif "," in s and "." in s:
            s = s.replace(",", "")
        return s
    normalized = orig.map(norm_one)
    return pd.to_numeric(normalized, errors="coerce")


def clean_df(df):
    df = df.copy().dropna(how="all").dropna(axis=1, how="all")
    if df.empty or len(df.columns) == 0:
        raise ValueError("В таблице не найдены данные. Проверь, что есть заголовки и строки.")
    df.columns = unique_columns(df.columns)
    for col in list(df.columns):
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            original = df[col]
            converted = coerce_number_series(original)
            nonempty = original.notna() & original.astype(str).str.strip().ne("")
            if nonempty.sum() and converted[nonempty].notna().sum() / nonempty.sum() >= 0.8:
                df[col] = converted
                continue
            # Parse dates only when column name or values look date-like and conversion is strong.
            col_hint = bool(re.search(r"date|дата|месяц|month|time|время|год|year|день", str(col), re.I))
            if col_hint or original.astype(str).str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", regex=True).mean() > 0.5:
                parsed = pd.to_datetime(original, errors="coerce", dayfirst=True)
                if nonempty.sum() and parsed[nonempty].notna().sum() / nonempty.sum() >= 0.7:
                    df[col] = parsed
    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS)
    if len(df.columns) > MAX_COLS:
        df = df.iloc[:, :MAX_COLS]
    return df.reset_index(drop=True)


def parse_text(text):
    text = (text or "").strip()
    if not text:
        raise ValueError("Пустое сообщение.")
    # Convert Markdown table rows; remove separator row like |---|---|.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2 and all("|" in line for line in lines[:2]):
        lines = [line for line in lines if not re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?", line)]
        text = "\n".join(line.strip().strip("|") for line in lines)
        df = pd.read_csv(io.StringIO(text), sep=r"\s*\|\s*", engine="python")
    else:
        # TSV pasted from Excel is common in Telegram.
        first = lines[0] if lines else text
        candidates = ["\t", ";", "|", ","]
        sep = max(candidates, key=lambda c: first.count(c))
        if first.count(sep) == 0:
            raise ValueError("Не вижу разделителей между столбцами. Скопируй таблицу вместе с заголовками.")
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
        except Exception:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    return clean_df(df)


def parse_file(filename, content):
    low = (filename or "table.csv").lower()
    if low.endswith((".xlsx", ".xlsm")):
        df = pd.read_excel(io.BytesIO(content), sheet_name=0)
    elif low.endswith((".csv", ".txt", ".tsv")):
        decoded = None
        for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
            try:
                decoded = content.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if decoded is None:
            raise ValueError("Не удалось прочитать файл. Сохрани CSV в UTF-8.")
        if low.endswith(".tsv"):
            df = pd.read_csv(io.StringIO(decoded), sep="\t")
            return clean_df(df)
        return parse_text(decoded)
    else:
        raise ValueError("Поддерживаются Excel (.xlsx, .xlsm), CSV, TSV и TXT.")
    return clean_df(df)


def groq_client():
    if not GROQ_KEY:
        raise RuntimeError("Добавь GROQ_API_KEY в переменные окружения FadeHost.")
    return Groq(api_key=GROQ_KEY, timeout=90.0, max_retries=1)


def extract_json(raw):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Модель не вернула таблицу в ожидаемом формате.")
    return json.loads(raw[start:end + 1])


def ocr_photo(content):
    """Read a photo using Groq's vision-capable Qwen model."""
    client = groq_client()
    image = Image.open(io.BytesIO(content))
    image = ImageOps.exif_transpose(image).convert("RGB")
    # Enhance readability while keeping payload modest.
    image.thumbnail((2200, 2200))
    image = ImageEnhance.Contrast(image).enhance(1.08)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    prompt = """Ты OCR-агент для таблиц. Внимательно перепиши таблицу с изображения, сохранив порядок строк и столбцов. Верни ТОЛЬКО JSON вида {\"columns\":[\"Месяц\",\"Продажи\"],\"rows\":[[\"Январь\",120],[\"Февраль\",150]]}. Не додумывай цифры и не исправляй данные по смыслу. Нечитаемые/пустые ячейки обозначай null. Числа верни числами, даты и подписи — строками. Если есть несколько таблиц, выбери самую крупную основную таблицу. Если таблица совсем нечитаема, верни {\"columns\":[],\"rows\":[]}."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
        ]}],
        temperature=0.1,
        max_completion_tokens=6000,
        response_format={"type": "json_object"},
    )
    data = extract_json(response.choices[0].message.content)
    if not data.get("columns") or not data.get("rows"):
        raise ValueError("Не удалось распознать таблицу. Попробуй фото сверху, при хорошем освещении, или обрежь изображение вокруг таблицы.")
    columns = data["columns"]
    rows = data["rows"]
    width = len(columns)
    rows = [(row + [None] * width)[:width] if isinstance(row, list) else [row] + [None] * (width - 1) for row in rows]
    return clean_df(pd.DataFrame(rows, columns=columns))


def numeric_cols(df):
    return [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def date_cols(df):
    return [str(c) for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def categorical_cols(df):
    n = numeric_cols(df)
    d = date_cols(df)
    return [str(c) for c in df.columns if str(c) not in n and str(c) not in d]


def recommend_chart(df):
    """Deterministic semantic fallback; does not rely on AI availability."""
    nums = numeric_cols(df)
    dates = date_cols(df)
    cats = categorical_cols(df)
    if len(nums) >= 2 and dates:
        return "line", f"Есть временной столбец «{dates[0]}» и несколько числовых показателей — линия лучше показывает динамику."
    if len(nums) >= 2 and not cats and len(df) >= 8:
        return "scatter", "Два и более числовых показателя без категорий: точечная диаграмма поможет увидеть взаимосвязь."
    if nums and cats:
        unique_n = df[cats[0]].nunique(dropna=True)
        if len(nums) == 1 and 2 <= unique_n <= 6 and len(df) <= 8:
            return "pie", "Небольшое число категорий и один числовой показатель — можно показать состав долей."
        if len(nums) >= 2:
            return "bar", "Есть категории и несколько числовых показателей — столбцы удобно сравнивают значения между категориями."
        if unique_n <= 30:
            return "bar", "Есть категории и числовой показатель — столбцы наглядно сравнят значения."
    if len(nums) >= 2:
        return "heatmap", "Несколько числовых показателей: тепловая карта покажет парные корреляции, но сама по себе не доказывает причинность."
    if len(nums) == 1:
        return "hist", "Один числовой показатель: гистограмма покажет распределение и возможную асимметрию."
    if dates and nums:
        return "line", "В таблице есть даты и числовой показатель — линия покажет изменение во времени."
    return "bar", "Не найден очевидный числовой шаблон. Столбцы — универсальный старт для сравнения строк."


def choose_columns(df):
    nums, dates, cats = numeric_cols(df), date_cols(df), categorical_cols(df)
    x = dates[0] if dates else (cats[0] if cats else None)
    ys = nums[:5]
    return x, ys


def chart_data_summary(df):
    out = {"rows": int(len(df)), "columns": []}
    for col in df.columns[:20]:
        s = df[col]
        item = {"name": str(col), "type": "date" if pd.api.types.is_datetime64_any_dtype(s) else ("numeric" if pd.api.types.is_numeric_dtype(s) else "category"), "missing": int(s.isna().sum()), "unique": int(s.nunique(dropna=True))}
        if item["type"] == "numeric":
            v = pd.to_numeric(s, errors="coerce").dropna()
            if not v.empty:
                item.update({"min": round(float(v.min()), 4), "max": round(float(v.max()), 4), "mean": round(float(v.mean()), 4), "median": round(float(v.median()), 4)})
        else:
            item["examples"] = [str(v)[:60] for v in s.dropna().astype(str).head(4).tolist()]
        out["columns"].append(item)
    return out


def ai_insights(df, kind, recommendation):
    """AI narrative is optional: local numerical summary is always available."""
    if not GROQ_KEY:
        return None
    summary = chart_data_summary(df)
    sample = df.head(12).replace({np.nan: None}).to_dict(orient="records")
    prompt = f"""Ты аналитик данных. Отвечай на русском. Ни в коем случае не выдумывай числа. Используй только сводку и пример строк ниже. Отличай корреляцию от причинности. Кратко: 3-6 пунктов с конкретными наблюдениями, качеством данных и одним практичным следующим шагом. Если данных недостаточно, прямо скажи. Тип графика: {NAMES.get(kind, kind)}. Причина автоподбора: {recommendation}. Сводка: {json.dumps(summary, ensure_ascii=False, default=str)}. Пример первых строк: {json.dumps(sample, ensure_ascii=False, default=str)}"""
    try:
        response = groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": "Ты точный аналитик данных. Не придумывай значения, не делай причинных выводов по корреляции."}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=700,
        )
        return (response.choices[0].message.content or "").strip()[:3500] or None
    except Exception:
        log.exception("Groq analysis unavailable; using local statistics")
        return None


def make_chart(df, kind):
    nums = numeric_cols(df)
    dates = date_cols(df)
    cats = categorical_cols(df)
    x = dates[0] if dates else (cats[0] if cats else None)
    ys = nums[:5]
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    title = NAMES.get(kind, kind)

    if kind == "hist":
        if not nums: raise ValueError("Для гистограммы нужен числовой столбец.")
        col = nums[0]
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty: raise ValueError("Нет числовых значений для гистограммы.")
        ax.hist(vals, bins=min(30, max(5, int(np.sqrt(len(vals))))), edgecolor="white")
        ax.set_xlabel(col); ax.set_ylabel("Количество"); ax.set_title(f"Распределение: {col}")

    elif kind == "box":
        if not nums: raise ValueError("Для диаграммы размаха нужен числовой столбец.")
        values = [pd.to_numeric(df[c], errors="coerce").dropna().values for c in nums[:8]]
        labels = nums[:8]
        ax.boxplot(values, labels=labels, showmeans=True)
        ax.set_title("Разброс и возможные выбросы"); ax.set_ylabel("Значение")
        if len(labels) > 3: ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=.25)

    elif kind == "heatmap":
        if len(nums) < 2: raise ValueError("Для тепловой карты нужны минимум два числовых столбца.")
        corr = df[nums[:12]].corr(numeric_only=True)
        im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=40, ha="right")
        ax.set_yticks(range(len(corr.index)), labels=corr.index)
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                val = corr.iloc[i, j]
                if pd.notna(val): ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Корреляция Пирсона")
        ax.set_title("Корреляции числовых показателей (не доказывают причинность)")

    elif kind == "scatter":
        if len(nums) < 2: raise ValueError("Для точечной диаграммы нужны два числовых столбца.")
        xx, yy = nums[0], nums[1]
        d = df[[xx, yy]].apply(pd.to_numeric, errors="coerce").dropna()
        if d.empty: raise ValueError("Нет пар числовых значений.")
        ax.scatter(d[xx], d[yy], alpha=.75, edgecolors="white", linewidths=.3)
        if len(d) >= 3 and d[xx].nunique() > 1:
            try:
                slope, intercept = np.polyfit(d[xx], d[yy], 1)
                xline = np.linspace(d[xx].min(), d[xx].max(), 100)
                ax.plot(xline, slope*xline+intercept, linestyle="--", linewidth=1.5, label="Линейный тренд")
                ax.legend()
            except Exception: pass
        ax.set_xlabel(xx); ax.set_ylabel(yy); ax.set_title(f"Связь: {yy} и {xx}"); ax.grid(True, alpha=.25)

    elif kind == "pie":
        if not nums: raise ValueError("Для круговой диаграммы нужен числовой столбец.")
        y = nums[0]
        if x is None or x == y:
            labels = [str(i + 1) for i in range(len(df))]
        else:
            labels = df[x].fillna("Без названия").astype(str).tolist()
        values = pd.to_numeric(df[y], errors="coerce").fillna(0)
        data = pd.DataFrame({"label": labels, "value": values}).groupby("label", as_index=False)["value"].sum()
        data = data[data.value > 0].sort_values("value", ascending=False)
        if data.empty: raise ValueError("Для круговой диаграммы нужны положительные значения.")
        if len(data) > 8:
            rest = data.iloc[7:]["value"].sum()
            data = pd.concat([data.iloc[:7], pd.DataFrame([{"label": "Прочее", "value": rest}])], ignore_index=True)
        ax.pie(data.value, labels=data.label, autopct="%1.1f%%", startangle=90, pctdistance=.75)
        ax.set_title(f"Доли: {y}")

    elif kind in ("line", "area"):
        if not nums: raise ValueError("Не найден числовой столбец для графика.")
        d = df.copy().head(500)
        if x:
            labels = d[x]
            if pd.api.types.is_datetime64_any_dtype(labels):
                order = labels.argsort(kind="stable")
                d = d.iloc[order]
                labels = d[x]
            labels = labels.astype(str).fillna("")
        else:
            labels = pd.Series(range(1, len(d) + 1), index=d.index)
        plotted = 0
        for col in nums:
            vals = pd.to_numeric(d[col], errors="coerce")
            if vals.notna().sum():
                if kind == "area": ax.fill_between(range(len(d)), vals.fillna(0).values, alpha=.18, label=col); ax.plot(range(len(d)), vals.values, linewidth=1.6)
                else: ax.plot(range(len(d)), vals.values, marker="o" if len(d) <= 40 else None, linewidth=1.8, label=col)
                plotted += 1
        if not plotted: raise ValueError("Нет числовых значений для построения.")
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(labels, rotation=40 if len(d) > 8 else 0, ha="right" if len(d) > 8 else "center")
        ax.set_xlabel(x or "Номер строки"); ax.set_ylabel("Значение"); ax.set_title(title)
        if len(nums) > 1: ax.legend(loc="best")
        ax.grid(True, alpha=.25)

    elif kind in ("bar", "stacked"):
        if not nums: raise ValueError("Для столбчатой диаграммы нужен числовой столбец.")
        if x:
            d = df[[x] + nums[:5]].copy()
            d = d.dropna(subset=[x]).head(30)
            d[x] = d[x].astype(str)
            # If too many categories, retain top 15 by first numeric series.
            if len(d) > 15:
                d = d.nlargest(15, nums[0])
            d = d.set_index(x)
        else:
            d = df[nums[:5]].head(20).copy()
            d.index = [str(i + 1) for i in range(len(d))]
        d = d.apply(pd.to_numeric, errors="coerce").fillna(0)
        if d.empty: raise ValueError("Нет данных для столбчатой диаграммы.")
        if kind == "stacked": d.plot(kind="bar", stacked=True, ax=ax)
        else: d.plot(kind="bar", ax=ax)
        ax.set_title(title); ax.set_xlabel(x or "Строка"); ax.set_ylabel("Значение")
        if len(d.index) > 8: ax.tick_params(axis="x", labelrotation=40)
        ax.grid(axis="y", alpha=.25)
        ax.legend(loc="best", fontsize=8)

    else:
        raise ValueError("Неизвестный тип графика.")

    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out.getvalue()


def local_analysis(df):
    lines = ["📋 Анализ данных", f"• Строк: {len(df)}; столбцов: {len(df.columns)}", "• Столбцы: " + ", ".join(map(str, df.columns[:15]))]
    missing = int(df.isna().sum().sum())
    if missing:
        lines.append(f"• Пустых ячеек: {missing} из {df.size} ({missing / max(df.size, 1):.1%}).")
    dup = int(df.duplicated().sum())
    if dup:
        lines.append(f"• Полностью повторяющихся строк: {dup}.")
    for col in numeric_cols(df)[:5]:
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        if not v.empty:
            lines.append(f"• {col}: среднее {v.mean():.2f}; медиана {v.median():.2f}; минимум {v.min():.2f}; максимум {v.max():.2f}.")
            if len(v) >= 3 and v.std(ddof=0) > 0:
                z = ((v - v.mean()) / v.std(ddof=0)).abs()
                outliers = int((z > 3).sum())
                if outliers: lines.append(f"  Возможных экстремальных значений (|z| > 3): {outliers}; проверь их перед выводами.")
    nums = numeric_cols(df)
    if len(nums) >= 2 and len(df) >= 3:
        corr = df[nums].corr(numeric_only=True)
        pairs = []
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                val = corr.iloc[i, j]
                if pd.notna(val): pairs.append((abs(float(val)), float(val), nums[i], nums[j]))
        if pairs:
            _, value, a, b = max(pairs)
            lines.append(f"• Самая сильная парная линейная корреляция среди числовых столбцов: {a} и {b} (r={value:.2f}). Корреляция не означает причинность.")
    if len(df) >= MAX_ROWS:
        lines.append(f"• Обработаны первые {MAX_ROWS} строк.")
    return "\n".join(lines)


def excel_with_chart(df, png, kind):
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Font, PatternFill
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Данные", index=False)
        ws = writer.sheets["Данные"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="34495E")
        for cells in ws.columns:
            width = max((len(str(c.value)) if c.value is not None else 0) for c in cells)
            ws.column_dimensions[cells[0].column_letter].width = min(max(width + 2, 12), 34)
        chart = writer.book.create_sheet("График")
        chart["A1"] = f"{NAMES.get(kind, kind)} — создано ботом"
        chart["A1"].font = Font(bold=True, size=14)
        img = XLImage(io.BytesIO(png)); img.width = 880; img.height = 520
        chart.add_image(img, "A3")
    out.seek(0)
    return out.getvalue()


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("df", None)
    await update.effective_message.reply_text(
        "Привет! Я ИИ-аналитик таблиц 📊\n\n"
        "Пришли Excel, CSV/TSV, фото или скопируй таблицу в сообщение. Я очищу данные, предложу график по смыслу и помогу сравнить другие варианты.\n\n"
        "На выходе: PNG, Excel с графиком и статистический анализ.\n\n/help — инструкция\n/cancel — сбросить таблицу"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "1) Отправь .xlsx/.xlsm, .csv/.tsv/.txt, фото или таблицу текстом.\n"
        "2) Нажми «Подобрать по смыслу» или выбери график вручную.\n"
        "3) Получи PNG, Excel и анализ.\n\n"
        "Виды: линия, столбцы, накопительные столбцы, области, связь X–Y, доли, распределение, размах, тепловая карта корреляций.\n"
        "Для чтения фото и ИИ-комментария нужен GROQ_API_KEY. Без него файлы и текстовые таблицы работают, а автоподбор и базовая статистика доступны локально."
    )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("df", None)
    await update.effective_message.reply_text("Текущая таблица сброшена. Отправь новые данные.")


async def accept_df(update, context, df):
    context.user_data["df"] = df
    preview = df.head(5).to_string(index=False, max_cols=8)
    if len(preview) > 1500: preview = preview[:1500] + "…"
    kind, why = recommend_chart(df)
    context.user_data["recommended_kind"] = kind
    context.user_data["recommendation_reason"] = why
    await update.effective_message.reply_text(
        f"✅ Таблица прочитана: {len(df)} строк, {len(df.columns)} столбцов.\n\n"
        f"Предлагаю: {NAMES[kind]}.\nПочему: {why}\n\nПервые строки:\n<pre>{esc(preview)}</pre>\n\n"
        "Нажми автоподбор или выбери другой график:",
        parse_mode="HTML", reply_markup=KEYBOARD
    )


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    doc = msg.document
    name = doc.file_name or "table.csv"
    if doc.file_size and doc.file_size > 15 * 1024 * 1024:
        await msg.reply_text("Файл слишком большой. Максимальный размер — 15 МБ.")
        return
    try:
        f = await doc.get_file()
        content = bytes(await f.download_as_bytearray())
        df = await asyncio.to_thread(parse_file, name, content)
        await accept_df(update, context, df)
    except Exception as e:
        log.exception("Ошибка чтения файла")
        await msg.reply_text(f"Не удалось прочитать файл: {str(e)[:350]}")


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not GROQ_KEY:
        await msg.reply_text("Для распознавания фото добавь GROQ_API_KEY в переменные окружения FadeHost. Excel, CSV и текстовые таблицы работают без API.")
        return
    try:
        f = await msg.photo[-1].get_file()
        content = bytes(await f.download_as_bytearray())
        await msg.reply_text("🔎 Читаю таблицу на фото. Проверь результат по предпросмотру — OCR может ошибаться в мелких цифрах…")
        df = await asyncio.to_thread(ocr_photo, content)
        await accept_df(update, context, df)
    except Exception as e:
        log.exception("Ошибка распознавания фото")
        await msg.reply_text(f"Не удалось распознать таблицу. Попробуй фото сверху, хорошее освещение и обрезку вокруг таблицы.\nПодробность: {str(e)[:250]}")


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    try:
        df = await asyncio.to_thread(parse_text, msg.text or "")
        if len(df.columns) < 2:
            await msg.reply_text("Нашёл меньше двух столбцов. Пришли заголовки и минимум два столбца, разделённых табуляцией, ; или |.")
            return
        await accept_df(update, context, df)
    except Exception:
        await msg.reply_text("Не получилось распознать сообщение как таблицу. Пример:\n<pre>Месяц;Продажи;Расходы\nЯнварь;120;70\nФевраль;150;80\nМарт;135;75</pre>", parse_mode="HTML")


async def chart_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    requested = q.data.split(":", 1)[1]
    df = context.user_data.get("df")
    if df is None:
        await q.message.reply_text("Сначала отправь таблицу.")
        return
    kind = requested
    why = context.user_data.get("recommendation_reason", "")
    if requested == "auto":
        kind, why = recommend_chart(df)
        context.user_data["recommended_kind"] = kind
        await q.message.reply_text(f"✨ Автоподбор: {NAMES[kind]}\n{why}")
    try:
        png = await asyncio.to_thread(make_chart, df, kind)
        xlsx = await asyncio.to_thread(excel_with_chart, df, png, kind)
        await q.message.reply_photo(photo=io.BytesIO(png), filename="grafik.png", caption=f"✅ {NAMES[kind]}. Строк: {len(df)}, столбцов: {len(df.columns)}.")
        await q.message.reply_document(document=io.BytesIO(xlsx), filename="таблица_и_график.xlsx", caption="Excel: исходные данные и график на отдельных листах.")
        report = local_analysis(df)
        ai = await asyncio.to_thread(ai_insights, df, kind, why)
        if ai:
            report += "\n\n🤖 Комментарий ИИ (проверь выводы по исходным данным):\n" + ai
        elif GROQ_KEY:
            report += "\n\nИИ-комментарий сейчас недоступен; выше приведена локальная статистика."
        await q.message.reply_text(report[:3900])
        await q.message.reply_text("Хочешь сравнить другой тип графика или прислать новую таблицу?", reply_markup=KEYBOARD)
    except Exception as e:
        log.exception("Ошибка построения графика")
        await q.message.reply_text(f"Не удалось построить график: {str(e)[:350]}")


def main():
    if not TOKEN:
        raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN в переменных окружения.")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(CallbackQueryHandler(chart_handler, pattern=r"^chart:(auto|line|bar|stacked|area|scatter|pie|hist|box|heatmap)$"))
    app.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    log.info("Продвинутый бот анализа таблиц запущен.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
