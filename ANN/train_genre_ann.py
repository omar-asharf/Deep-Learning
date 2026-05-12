from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


ANN_DIR = Path(__file__).resolve().parent
DATA_FILE = ANN_DIR / "data" / "BooksClassifier_dataset_cleaned.csv.xls"
MODEL_DIR = ANN_DIR / "model"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
SELECTION_METRIC = "accuracy"


@dataclass(frozen=True)
class Experiment:
    hidden_layer_sizes: tuple[int, ...]
    feature_builder: str
    alpha: float
    sample_weight: str | None = None


def clean_text_column(text: pd.Series) -> pd.Series:
    return text.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def build_label_pattern(labels: list[str]) -> re.Pattern[str]:
    label_phrases = sorted({label.lower() for label in labels}, key=len, reverse=True)
    escaped = [re.escape(label).replace(r"\ ", r"\s+") for label in label_phrases]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


def load_data(data_file: Path) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    df = pd.read_csv(data_file)
    initial_rows = len(df)

    df = df[["cleaned_text", "target_genre"]].dropna().copy()
    rows_after_dropna = len(df)
    df["cleaned_text"] = clean_text_column(df["cleaned_text"])
    df = df[df["cleaned_text"] != ""].copy()
    rows_after_empty_text = len(df)

    exact_duplicate_rows = int(df.duplicated(subset=["cleaned_text", "target_genre"]).sum())
    duplicate_text_rows = int(df.duplicated(subset=["cleaned_text"]).sum())
    conflicting_duplicate_texts = int(
        (df.groupby("cleaned_text")["target_genre"].nunique() > 1).sum()
    )

    df = df.drop_duplicates(subset=["cleaned_text"], keep="first").reset_index(drop=True)

    class_names = sorted(df["target_genre"].unique())
    label_pattern = build_label_pattern(class_names)
    rows_with_label_terms = int(df["cleaned_text"].str.contains(label_pattern, regex=True).sum())

    audit = {
        "initial_rows": int(initial_rows),
        "rows_after_dropna": int(rows_after_dropna),
        "rows_after_empty_text_filter": int(rows_after_empty_text),
        "rows_after_duplicate_text_filter": int(len(df)),
        "exact_duplicate_rows_removed": exact_duplicate_rows,
        "duplicate_text_rows_removed": duplicate_text_rows,
        "conflicting_duplicate_texts": conflicting_duplicate_texts,
        "rows_with_class_name_terms": rows_with_label_terms,
    }
    return df["cleaned_text"], df["target_genre"], audit


def build_features(kind: str) -> FeatureUnion | TfidfVectorizer:
    if kind == "word":
        return TfidfVectorizer(
            max_features=3000,
            min_df=2,
            max_df=0.90,
            sublinear_tf=True,
        )

    if kind == "word_bigram":
        return TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.90,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

    if kind == "word_char":
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        max_features=16000,
                        min_df=2,
                        max_df=0.90,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=4000,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown feature builder: {kind}")


def build_ann(experiment: Experiment) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", build_features(experiment.feature_builder)),
            (
                "ann",
                MLPClassifier(
                    hidden_layer_sizes=experiment.hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=experiment.alpha,
                    batch_size=64,
                    learning_rate_init=0.001,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=6,
                    max_iter=70,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def score_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
    }


def top_confusions(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> list[dict[str, object]]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    confusions: list[dict[str, object]] = []

    for true_index, true_label in enumerate(labels):
        for pred_index, pred_label in enumerate(labels):
            if true_index == pred_index:
                continue

            count = int(matrix[true_index, pred_index])
            if count:
                confusions.append(
                    {
                        "true_genre": true_label,
                        "predicted_genre": pred_label,
                        "count": count,
                    }
                )

    return sorted(confusions, key=lambda item: item["count"], reverse=True)[:10]


def fit_model(
    model: Pipeline,
    experiment: Experiment,
    x_train: pd.Series,
    y_train: pd.Series,
) -> None:
    if experiment.sample_weight == "balanced":
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(x_train, y_train, ann__sample_weight=weights)
        return

    model.fit(x_train, y_train)


def train_and_evaluate(
    data_file: Path,
    model_dir: Path,
    selection_metric: str,
) -> dict[str, object]:
    texts, labels, audit = load_data(data_file)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        texts,
        encoded_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=encoded_labels,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    experiments = {
        "ANN_64_unigram": Experiment(
            hidden_layer_sizes=(64,),
            feature_builder="word",
            alpha=0.0001,
        ),
        "ANN_128_word_bigram": Experiment(
            hidden_layer_sizes=(128,),
            feature_builder="word_bigram",
            alpha=0.0003,
        ),
        "ANN_128_word_char": Experiment(
            hidden_layer_sizes=(128,),
            feature_builder="word_char",
            alpha=0.0005,
        ),
        "ANN_128_word_char_balanced": Experiment(
            hidden_layer_sizes=(128,),
            feature_builder="word_char",
            alpha=0.0005,
            sample_weight="balanced",
        ),
    }

    class_names = list(label_encoder.classes_)
    results: dict[str, object] = {
        "data_file": str(data_file),
        "random_state": RANDOM_STATE,
        "selection_metric": selection_metric,
        "data_checks": audit,
        "rows_used": int(len(texts)),
        "classes": int(labels.nunique()),
        "class_names": class_names,
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_validation)),
        "train_validation_rows": int(len(x_train_val)),
        "test_rows": int(len(x_test)),
        "experiments": {},
    }

    best_name = ""
    best_experiment: Experiment | None = None
    best_metric = -1.0

    y_validation_names = label_encoder.inverse_transform(y_validation)
    for name, experiment in experiments.items():
        model = build_ann(experiment)
        fit_model(model, experiment, x_train, y_train)
        predictions = label_encoder.inverse_transform(model.predict(x_validation))
        scores = score_predictions(y_validation_names, predictions)

        results["experiments"][name] = {
            "hidden_layers": experiment.hidden_layer_sizes,
            "feature_builder": experiment.feature_builder,
            "alpha": experiment.alpha,
            "sample_weight": experiment.sample_weight,
            "validation": scores,
            "ann_iterations": int(model.named_steps["ann"].n_iter_),
        }

        if scores[selection_metric] > best_metric:
            best_metric = scores[selection_metric]
            best_name = name
            best_experiment = experiment

    assert best_experiment is not None
    best_model = build_ann(best_experiment)
    fit_model(best_model, best_experiment, x_train_val, y_train_val)

    best_predictions = label_encoder.inverse_transform(best_model.predict(x_test))
    y_test_names = label_encoder.inverse_transform(y_test)
    test_scores = score_predictions(y_test_names, best_predictions)

    results["best_model"] = best_name
    results["best_validation_score"] = best_metric
    results["test"] = test_scores
    results["classification_report"] = classification_report(
        y_test_names,
        best_predictions,
        output_dict=True,
        zero_division=0,
    )
    results["top_confusions"] = top_confusions(y_test_names, pd.Series(best_predictions), class_names)

    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "genre_ann_model.joblib"
    metrics_path = model_dir / "metrics.json"
    joblib.dump(
        {
            "model": best_model,
            "label_encoder": label_encoder,
            "best_model": best_name,
            "selection_metric": selection_metric,
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Rows used: {results['rows_used']}")
    print(f"Classes: {results['classes']}")
    print(f"Selection metric: {selection_metric}")
    print("Data checks:")
    print(f"  Duplicate text rows removed: {audit['duplicate_text_rows_removed']}")
    print(f"  Conflicting duplicate texts: {audit['conflicting_duplicate_texts']}")
    print(f"  Rows containing class-name terms: {audit['rows_with_class_name_terms']}")
    print("Validation results:")
    for name, details in results["experiments"].items():
        scores = details["validation"]
        print(
            f"  {name}: accuracy={scores['accuracy']}, "
            f"macro_f1={scores['macro_f1']}, weighted_f1={scores['weighted_f1']}"
        )
    print(
        f"Final test: accuracy={test_scores['accuracy']}, "
        f"macro_f1={test_scores['macro_f1']}, weighted_f1={test_scores['weighted_f1']}"
    )
    print(f"Best model: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")

    return results


def predict_text(model_path: Path, text: str) -> None:
    saved = joblib.load(model_path)
    model = saved["model"]
    label_encoder = saved["label_encoder"]
    prediction = label_encoder.inverse_transform(model.predict([text]))[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        classes = label_encoder.inverse_transform(model.classes_)
        ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
        print(f"Prediction: {prediction}")
        print("Top probabilities:")
        for genre, probability in ranked[:3]:
            print(f"  {genre}: {probability:.3f}")
    else:
        print(f"Prediction: {prediction}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ANN book genre classifier.")
    parser.add_argument("--data", type=Path, default=DATA_FILE, help="Cleaned CSV data file.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="Output folder.")
    parser.add_argument(
        "--selection-metric",
        choices=["accuracy", "macro_f1", "weighted_f1"],
        default=SELECTION_METRIC,
        help="Validation metric used to choose the final ANN before test evaluation.",
    )
    parser.add_argument("--predict", type=str, help="Predict a genre for one text sample.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.predict:
        predict_text(args.model_dir / "genre_ann_model.joblib", args.predict)
        return

    train_and_evaluate(
        data_file=args.data,
        model_dir=args.model_dir,
        selection_metric=args.selection_metric,
    )


if __name__ == "__main__":
    main()
