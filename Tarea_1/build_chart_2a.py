"""Genera informe/chart_2a.png combinando: top-20 ingenuo, top-20 filtrado y heatmap.

Reproduce las celdas relevantes del notebook 2026_tarea1.ipynb (sección 2.A).
"""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "informe" / "chart_2a.png"

try:
    from nltk.corpus import stopwords as _stop
    _ = _stop.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as _nltk_stopwords  # noqa: E402

STOPWORDS_EN = set(_nltk_stopwords.words("english"))

MEDIA_IDENTIFIERS = {
    "Reuters":            {"reuters", "reporting", "editing"},
    "The Hill":           {"thehill"},
    "CNBC":               {"cnbc", "madcap"},
    "The New York Times": {"nytimes", "nyt", "nytopinion"},
    "People":             {"people"},
}
ALL_MEDIA_TOKENS = set().union(*MEDIA_IDENTIFIERS.values())
EXTRA_STOPWORDS = STOPWORDS_EN | ALL_MEDIA_TOKENS

MEDIA_BOILERPLATE_PATTERNS = [
    r"(?i)\b[\w\s.,'-]{0,60}?\(reuters\)\s*[-–—]\s*",
    r"(?i)reporting by [^.\n]{0,200}",
    r"(?i)editing by [^.\n]{0,200}",
    r"(?i)further company coverage[^\n]{0,500}",
    r"(?i)\|\s*thehill\b",
    r"(?i)capitol hill publishing corp[^\n]{0,500}",
    r"(?i)madcap@cnbc\.com",
    r"@NYTopinion",
]


def clean_text(df, column_name):
    result = df[column_name].str.replace(r"^[^\n]*\n", "", regex=True)
    result = result.str.lower()
    result = result.str.replace(r"https?://\S+", " ", regex=True)
    result = result.str.replace(r"[^\w\s]", " ", regex=True)
    result = result.str.replace(r"\b\d+\b", " ", regex=True)
    result = result.str.replace(r"[\n\r\t]", " ", regex=True)
    result = result.str.replace(r"\s+", " ", regex=True)
    return result.str.strip()


def clean_text_v2(df, column_name,
                  extra_stopwords=EXTRA_STOPWORDS,
                  boilerplate_patterns=MEDIA_BOILERPLATE_PATTERNS):
    s = df[column_name].astype(object)
    for pattern in boilerplate_patterns:
        s = s.str.replace(pattern, " ", regex=True)
    df_pre = df.copy()
    df_pre[column_name] = s
    cleaned = clean_text(df_pre, column_name)
    return cleaned.apply(
        lambda x: [t for t in x.split() if t and t not in extra_stopwords]
        if isinstance(x, str) else []
    )


def top_words_naive(df, top_5, top_n=20):
    out = {}
    for pub in top_5:
        mask = df["publication"] == pub
        counter = Counter()
        for text in df.loc[mask, "CleanText"].dropna():
            counter.update(text.split())
        total = sum(counter.values()) or 1
        out[pub] = [(w, c / total * 100) for w, c in counter.most_common(top_n)]
    return out


def top_words_filtered(df, top_5, top_n=20):
    out = {}
    for pub in top_5:
        mask = df["publication"] == pub
        counter = Counter()
        for tokens in df.loc[mask, "Tokens"]:
            counter.update(tokens)
        total = sum(counter.values()) or 1
        out[pub] = [(w, c / total * 100) for w, c in counter.most_common(top_n)]
    return out


def _draw_bar_panel(parent_axes, top_words, palette):
    for i, (pub, items) in enumerate(top_words.items()):
        ax = parent_axes[i]
        words, freqs = zip(*items)
        ax.barh(words[::-1], freqs[::-1], color=palette[i % len(palette)])
        ax.set_title(pub, fontsize=11)
        ax.set_xlabel("Frecuencia relativa (%)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", alpha=0.3)
    for j in range(len(top_words), len(parent_axes)):
        parent_axes[j].axis("off")


def main():
    print("[1/5] Cargando dataset (cache local)...")
    ds = load_dataset(
        "tomas-gr/all-the-news-2-1-Component-one-sampled",
        split="train",
        cache_dir=str(HERE / "data"),
    )
    df = ds.to_pandas()

    print("[2/5] Filtrando top-5 medios y aplicando clean_text / clean_text_v2...")
    top_5 = df["publication"].value_counts().head(5).index.tolist()
    df_top_5 = df[df["publication"].isin(top_5)].copy()
    df_top_5["CleanText"] = clean_text(df_top_5, "article")
    df_top_5["Tokens"] = clean_text_v2(df_top_5, "article")

    print("[3/5] Calculando top-20 ingenuo y filtrado...")
    top_naive = top_words_naive(df_top_5, top_5, top_n=20)
    top_clean = top_words_filtered(df_top_5, top_5, top_n=20)

    print("[4/5] Construyendo heatmap...")
    TOP_N_HEAT = 10
    all_counters = {}
    for pub in top_5:
        counter = Counter()
        for tokens in df_top_5.loc[df_top_5["publication"] == pub, "Tokens"]:
            counter.update(tokens)
        all_counters[pub] = (counter, sum(counter.values()) or 1)

    distinct_words = set()
    for pub, items in top_clean.items():
        for w, _ in items[:TOP_N_HEAT]:
            distinct_words.add(w)

    heat_df = pd.DataFrame(
        {pub: {w: all_counters[pub][0].get(w, 0) / all_counters[pub][1] * 100
               for w in distinct_words}
         for pub in top_5}
    )
    heat_df = heat_df.loc[
        (heat_df.max(axis=1) - heat_df.min(axis=1))
        .sort_values(ascending=False).index
    ]

    print("[5/5] Renderizando figura combinada y guardando PNG...")
    palette = sns.color_palette("tab10", n_colors=len(top_5))

    fig = plt.figure(figsize=(18, 36))
    gs = GridSpec(
        nrows=7, ncols=3, figure=fig,
        height_ratios=[0.05, 1, 1, 0.05, 1, 1, 1.5],
        hspace=0.55, wspace=0.35,
    )

    ax_t1 = fig.add_subplot(gs[0, :]); ax_t1.axis("off")
    ax_t1.text(0.5, 0.5,
               "Versión ingenua — top 20 palabras por medio sobre CleanText "
               "(sin filtrar stopwords ni pistas del medio)",
               ha="center", va="center", fontsize=14, fontweight="bold")

    naive_axes = [fig.add_subplot(gs[1, c]) for c in range(3)] + \
                 [fig.add_subplot(gs[2, c]) for c in range(3)]
    _draw_bar_panel(naive_axes, top_naive, palette)

    ax_t2 = fig.add_subplot(gs[3, :]); ax_t2.axis("off")
    ax_t2.text(0.5, 0.5,
               "Versión filtrada — top 20 palabras por medio sobre Tokens "
               "(sin stopwords ni pistas del medio)",
               ha="center", va="center", fontsize=14, fontweight="bold")

    filt_axes = [fig.add_subplot(gs[4, c]) for c in range(3)] + \
                [fig.add_subplot(gs[5, c]) for c in range(3)]
    _draw_bar_panel(filt_axes, top_clean, palette)

    ax_heat = fig.add_subplot(gs[6, :])
    sns.heatmap(
        heat_df, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax_heat,
        cbar_kws={"label": "Frecuencia relativa (%)"},
    )
    ax_heat.set_title(
        "Heatmap — frecuencia relativa (%) de las top 10 palabras de cada medio "
        "(versión filtrada)",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax_heat.set_xlabel("Medio")
    ax_heat.set_ylabel("Palabra")

    fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"OK -> {OUT_PATH}")


if __name__ == "__main__":
    main()
