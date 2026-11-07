import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_curve,
)

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Качество вина", page_icon="🍷", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(BASE / "model.joblib")


@st.cache_data
def load_test():
    return pd.read_csv(BASE / "test.csv")


@st.cache_data
def probabilities(_model, X):
    # _model исключён из хеширования; X определяет набор предсказаний.
    class_one = list(_model.classes_).index(1)
    return _model.predict_proba(X)[:, class_one]


required = ["model.joblib", "config.json", "test.csv"]
missing = [name for name in required if not (BASE / name).is_file()]
if missing:
    st.error("Сначала выполните экспорт из части 3. Не найдены: " + ", ".join(missing))
    st.stop()

model = load_model()
test = load_test()
features = test.columns.drop("target").tolist()
X_test = test.loc[:, features]
y_test = test["target"]
# На повторных запусках, включая смену порога, используется кеш.
test_probability = probabilities(model, X_test)
default_threshold = float(json.loads((BASE / "config.json").read_text())["threshold"])
threshold = st.sidebar.slider(
    "Порог хорошего вина", 0.0, 1.0, value=default_threshold,
    step=0.001, format="%.6f", key="threshold",
)
st.sidebar.caption(f"Сохранённый порог: {default_threshold:.9g}. Класс 1 при p ≥ порога.")
st.title("🍷 Хорошее ли вино?")
single_tab, metrics_tab, csv_tab, weights_tab = st.tabs(
    ["Одно вино", "Качество модели", "CSV-прогноз", "Коэффициенты"]
)

with single_tab:
    st.write("Диапазоны и значения по умолчанию взяты из тестовой выборки.")
    values = {}
    columns = st.columns(3)
    for i, name in enumerate(features):
        with columns[i % 3]:
            if name == "is_red":
                values[name] = int(st.toggle(
                    "Красное вино (выключено - белое)",
                    value=bool(X_test[name].median()),
                ))
            else:
                values[name] = st.number_input(
                    name, min_value=float(X_test[name].min()),
                    max_value=float(X_test[name].max()),
                    value=float(X_test[name].median()),
                    step=0.001, format="%.6f",
                )
    wine = pd.DataFrame([values], columns=features)
    p = float(probabilities(model, wine)[0])
    st.metric("Вероятность хорошего вина", f"{p:.1%}")
    st.success("Хорошее вино (класс 1)" if p >= threshold else "Обычное вино (класс 0)")
    st.caption(f"Используется общий порог {threshold:.6f}.")

with metrics_tab:
    st.info(
        "Для сомелье: чем выше порог, тем меньше вин попадает в подборку. "
        "Обычно это повышает precision, но снижает recall: меньше обычных вин, "
        "зато часть хороших вин будет пропущена. При снижении порога подборка "
        "расширяется, recall растёт или остаётся прежним, но может стать больше "
        "ложноположительных результатов. Precision не обязан меняться монотонно."
    )
    labels = (test_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, labels, labels=[0, 1]).ravel()
    precision = precision_score(y_test, labels, zero_division=0)
    recall = recall_score(y_test, labels, zero_division=0)
    f1 = f1_score(y_test, labels, zero_division=0)
    for column, name, value in zip(
        st.columns(3), ["Precision", "Recall", "F1"], [precision, recall, f1]
    ):
        column.metric(name, f"{value:.3f}")
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay(
        np.array([[tn, fp], [fn, tp]]), display_labels=["Обычное", "Хорошее"]
    ).plot(ax=ax, colorbar=False)
    ax.set_title(f"Матрица ошибок, порог {threshold:.3f}")
    st.pyplot(fig)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, test_probability)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, test_probability)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr, label="ROC")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].scatter(fp / (fp + tn) if fp + tn else 0, recall,
                    color="red", label=f"Порог {threshold:.3f}", zorder=3)
    axes[0].set(xlabel="False positive rate", ylabel="Recall", title="ROC")
    axes[1].plot(pr_recall, pr_precision, label="PR")
    axes[1].axhline(y_test.mean(), linestyle="--", color="grey", label="Случайная модель")
    axes[1].scatter(recall, precision, color="red", label=f"Порог {threshold:.3f}", zorder=3)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall")
    for ax in axes:
        ax.legend()
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("При смене порога меняются метки, метрики и точки; вероятности взяты из кеша.")

with csv_tab:
    st.write("Загрузите CSV с теми же 12 признаками. Порядок столбцов не важен.")
    uploaded = st.file_uploader("CSV с винами", type=["csv"])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            absent = [name for name in features if name not in batch.columns]
            if absent:
                st.error("В CSV отсутствуют признаки: " + ", ".join(absent))
            elif batch.empty:
                st.error("CSV должен содержать хотя бы одну строку.")
            else:
                numeric = batch.loc[:, features].apply(pd.to_numeric, errors="raise")
                if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                    st.error("Признаки должны быть конечными числами без пропусков.")
                elif not numeric["is_red"].isin([0, 1]).all():
                    st.error("is_red должен равняться 0 (белое) или 1 (красное).")
                else:
                    p_batch = probabilities(model, numeric)
                    result = batch.copy()
                    result["probability"] = p_batch
                    result["prediction"] = (p_batch >= threshold).astype(int)
                    st.caption(f"Общий порог: {threshold:.6f}")
                    st.dataframe(result, hide_index=True)
                    st.download_button(
                        "Скачать прогнозы", result.to_csv(index=False).encode("utf-8"),
                        file_name="wine_predictions.csv", mime="text/csv",
                    )
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
            st.error(f"Не удалось прочитать CSV с числовыми признаками: {error}")

with weights_tab:
    classifier = model.steps[-1][1]
    weights = pd.Series(classifier.coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(weights.index, weights.values,
            color=["#b33d4b" if value < 0 else "#2b8c6d" for value in weights])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Коэффициент после StandardScaler", title="Вклад признаков в log-odds класса 1")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    positive = weights[weights > 0].nlargest(3)
    negative = weights[weights < 0].nsmallest(3)
    st.write("Сильнее повышают вероятность:", ", ".join(positive.index) or "нет")
    st.write("Сильнее понижают вероятность:", ", ".join(negative.index) or "нет")
    st.caption(
        "Знак показывает направление при прочих равных. После стандартизации "
        "веса сравнимы по масштабу; это связи в модели, а не доказательство причинности. "
        "Свободный член в график не включён."
    )
