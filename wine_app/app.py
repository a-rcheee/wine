import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

app_dir = Path(__file__).parent

@st.cache_resource
def load_model():
    return joblib.load(app_dir / 'model.joblib')

model = load_model()

with open(app_dir / 'config.json', encoding='utf-8') as f:
    default_threshold = json.load(f)['threshold']

@st.cache_data
def load_test_data():
    return pd.read_csv(app_dir / 'test.csv')

@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]

df = load_test_data()
X = df.drop(columns='target')

st.title('Рекомендация вина')

tab1, tab2, tab3, tab4 = st.tabs(['Оценка одного вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

with tab1:
    st.subheader('Характеристики вина')

    values = {}

    for col in X.columns:
        if col == 'is_red':
            values[col] = int(st.toggle(
                'Красное вино',
                value=bool(X[col].median())
            ))
        else:
            values[col] = st.number_input(
                col,
                min_value=float(X[col].min()),
                max_value=float(X[col].max()),
                value=float(X[col].median()),
                step=0.1
            )

    threshold = st.slider(
        'Порог классификации',
        min_value=0.0,
        max_value=1.0,
        value=float(default_threshold),
        step=0.001
    )

    wine = pd.DataFrame([values], columns=X.columns)
    probability = model.predict_proba(wine)[0, 1]

    prediction = int(probability >= threshold)

    st.metric('Вероятность хорошего вина', f'{probability:.2%}')

    if prediction == 1:
        st.success('Рекомендуем это вино!')
    else:
        st.warning('Не рекомендуем это вино.')

    st.write(f'Вероятность: {probability:.4f}')
    st.write(f'Выбранный порог: {threshold:.3f}')

import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score)

with tab2:
    st.subheader('Модельные метрики')

    threshold_metrics = st.slider(
        'Порог классификации',
        0.0, 1.0,
        value=float(default_threshold),
        step=0.001,
        key='metrics_threshold'
    )

    y = df['target']
    proba = test_proba(model, X)
    pred = (proba >= threshold_metrics).astype(int)

    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)

    col1, col2, col3 = st.columns(3)
    col1.metric('Precision', f'{precision:.3f}')
    col2.metric('Recall', f'{recall:.3f}')
    col3.metric('F1', f'{f1:.3f}')

    st.subheader('Матрица ошибок')

    cm = confusion_matrix(y, pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[0, 1]
    ).plot(ax=ax, cmap='Blues', colorbar=False)

    st.pyplot(fig)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y, proba)
    pr, rec, _ = precision_recall_curve(y, proba)

    roc_auc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)

    tn, fp, fn, tp = cm.ravel()

    fpr_point = fp / (fp + tn)
    tpr_point = recall

    st.subheader('ROC и PR-кривые')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].scatter( fpr_point, tpr_point, color='red', s=80,  label=f'Порог = {threshold_metrics:.3f}')
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC-кривая')
    axes[0].legend()

    axes[1].plot(rec, pr, label=f'PR-AUC = {pr_auc:.3f}')
    axes[1].axhline(y.mean(), color='gray', linestyle='--', label='Случайная модель')
    axes[1].scatter(recall, precision, color='red', s=80, label=f'Порог = {threshold_metrics:.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('PR-кривая')
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader('Как порог влияет на подборку?')

    st.write(
        'Если увеличить порог, модель будет рекомендовать меньше вин. '
        'Рекомендации обычно станут точнее, но часть хороших вин будет пропущена. '
        'Если уменьшить порог, в подборку попадёт больше хороших вин, '
        'но возрастёт риск рекомендовать плохое вино.'
    )
from io import BytesIO

with tab3:
    st.subheader('Пакетное предсказание вин')

    uploaded_file = st.file_uploader(
        'Загрузите CSV-файл с характеристиками вин',
        type='csv'
    )

    threshold_batch = st.session_state.get(
        'metrics_threshold', float(default_threshold)
    )

    st.write(f'Текущий порог: {threshold_batch:.3f}')

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            missing_cols = [col for col in X.columns if col not in uploaded_df.columns]

            if missing_cols:
                st.error('В файле отсутствуют необходимые столбцы: '  + ', '.join(missing_cols))
            else:
                X_uploaded = uploaded_df[X.columns]
                probabilities = model.predict_proba(X_uploaded)[:, 1]
                predictions = (probabilities >= threshold_batch).astype(int)

                result_df = uploaded_df.copy()
                result_df['probability'] = probabilities
                result_df['prediction'] = predictions

                st.success(f'Обработано вин: {len(result_df)}')

                st.dataframe(result_df)

                csv = result_df.to_csv(index=False).encode('utf-8-sig')

                st.download_button(label='Скачать результаты CSV', data=csv, file_name='wine_predictions.csv', mime='text/csv')

        except Exception as e:
            st.error(f'Ошибка при обработке файла: {e}')

with tab4:
    st.subheader('Коэффициенты логистической регрессии')

    coefficients = model.named_steps['logisticregression'].coef_[0]

    coef_df = pd.DataFrame({'Признак': X.columns, 'Коэффициент': coefficients}).sort_values('Коэффициент')

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ['tomato' if coef < 0 else 'seagreen' for coef in coef_df['Коэффициент']]

    ax.barh(coef_df['Признак'], coef_df['Коэффициент'],color=colors)

    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Коэффициент после масштабирования')
    ax.set_title('Влияние признаков на предсказание')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        'Положительные коэффициенты увеличивают вероятность '
        'хорошего вина, а отрицательные — уменьшают её. '
        'Чем больше коэффициент по модулю, тем сильнее влияние '
        'признака при прочих равных.'
    )

    st.write('**Сильнее всего за хорошее вино:**')
    st.write(', '.join(coef_df.tail(3)['Признак'].iloc[::-1]))

    st.write('**Сильнее всего против хорошего вина:**')
    st.write(', '.join(coef_df.head(3)['Признак']))
