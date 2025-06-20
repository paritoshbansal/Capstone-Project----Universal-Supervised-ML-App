
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression, HuberRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, confusion_matrix,
    classification_report, roc_curve, auc
)
from sklearn.base import BaseEstimator, TransformerMixin

# === Custom Transformer ===
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
    def fit(self, X, y=None):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_bounds_ = Q1 - self.factor * IQR
        self.upper_bounds_ = Q3 + self.factor * IQR
        return self
    def transform(self, X):
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

def is_tree_model(model_cls):
    return model_cls in [DecisionTreeRegressor, RandomForestRegressor, DecisionTreeClassifier, RandomForestClassifier]

# === Utility to convert plot to Streamlit ===
def get_fig_as_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

# === Main training + evaluation function ===
def train_and_evaluate(df, dependent_variable, future_df=None):
    output_plots = []
    result_text = ""

    x = df.drop(columns=[dependent_variable])
    y = df[dependent_variable]

    if y.nunique() <= 20 and y.dtype in ['int64', 'int32', 'object', 'category', 'bool']:
        task = "classification"
    else:
        task = "regression"

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    label_encoder = None
    if y_train.dtype in ['object', 'category', 'bool']:
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

    numeric_features = x.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = x.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('capper', OutlierCapper()),
        ('scaler', MinMaxScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    if task == "regression":
        models = [HuberRegressor, LinearRegression, KNeighborsRegressor, Lasso, Ridge, DecisionTreeRegressor, RandomForestRegressor]
        metric_func = r2_score
        model_key = "regressor"
    else:
        models = [SVC, LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]
        metric_func = accuracy_score
        model_key = "classifier"

    best_score = -1
    best_model = None
    best_model_name = ""
    y_transformer = None

    for model_cls in models:
        if task == "regression" and not is_tree_model(model_cls):
            y_transformer = PowerTransformer()
            y_train_trans = y_transformer.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_test_trans = y_transformer.transform(y_test.values.reshape(-1, 1)).flatten()
        else:
            y_train_trans = y_train
            y_test_trans = y_test
            y_transformer = None

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            (model_key, model_cls(probability=True) if model_cls == SVC else model_cls())
        ])
        try:
            pipe.fit(x_train, y_train_trans)
            y_pred = pipe.predict(x_test)
            score = metric_func(y_test_trans, y_pred)

            if score != 1 and score > best_score:
                best_model = pipe
                best_model_name = model_cls.__name__
                best_score = score
        except:
            continue

    # Try Naive Bayes (for classification only)
    if task == "classification":
        nb_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GaussianNB())
        ])
        nb_pipe.fit(x_train, y_train)
        y_pred = nb_pipe.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        if score > best_score:
            best_model = nb_pipe
            best_model_name = "GaussianNB"
            best_score = score

    if best_model is None:
        return "No suitable model found", [], None

    y_pred = best_model.predict(x_test)

    # === Results and plots ===
    if task == "classification":
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {best_model_name}')
        output_plots.append(get_fig_as_image(fig))

        report = classification_report(y_test, y_pred)
        result_text = f"Model: {best_model_name}\nAccuracy: {best_score:.4f}\n\n{report}"
    else:
        if y_transformer:
            y_pred = y_transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_test = y_transformer.inverse_transform(y_test_trans.reshape(-1, 1)).flatten()

        residuals = y_test - y_pred
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_test, y_pred, alpha=0.6)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        ax1.set_title("Actual vs Predicted")
        output_plots.append(get_fig_as_image(fig1))

        fig2, ax2 = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.set_title("Residuals Distribution")
        output_plots.append(get_fig_as_image(fig2))

        result_text = f"Model: {best_model_name}\nR² Score: {best_score:.4f}\nMAE: {mean_absolute_error(y_test, y_pred):.4f}\nMSE: {mean_squared_error(y_test, y_pred):.4f}"

    # === Future Predictions ===
    predictions_df = None
    if future_df is not None:
        try:
            future_df = future_df[x_train.columns]
            preds = best_model.predict(future_df)
            if y_transformer:
                preds = y_transformer.inverse_transform(preds.reshape(-1, 1)).flatten()
            if label_encoder:
                preds = label_encoder.inverse_transform(preds)
            future_df[f"Predicted_{dependent_variable}"] = preds
            predictions_df = future_df
        except Exception as e:
            result_text += f"\n\n(Failed to predict on future data: {e})"

    return result_text, output_plots, predictions_df
