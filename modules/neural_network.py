"""
Neural network AQI predictor with safe fallback mode.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


class AQINeuralNetwork:
    """Train and infer AQI category from pollutant features."""

    FEATURE_ORDER = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "hour", "is_peak"]
    CLASS_NAMES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

    def __init__(self, model_path: str = "models/nn_model.h5", scaler_path: str = "models/scaler.pkl") -> None:
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.sklearn_model_path = "models/nn_model_sklearn.pkl"
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("outputs").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _try_import_tf():
        try:
            import tensorflow as tf  # noqa: PLC0415
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # noqa: PLC0415
            from tensorflow.keras.layers import Dense, Dropout  # noqa: PLC0415
            from tensorflow.keras.models import Sequential, load_model  # noqa: PLC0415
            from tensorflow.keras.optimizers import Adam  # noqa: PLC0415

            return tf, Sequential, Dense, Dropout, Adam, EarlyStopping, ModelCheckpoint, load_model
        except Exception:
            return None

    @staticmethod
    def _rule_based_label(pm25: float) -> str:
        if pm25 < 30:
            return "Good"
        if pm25 < 60:
            return "Satisfactory"
        if pm25 < 90:
            return "Moderate"
        if pm25 < 120:
            return "Poor"
        if pm25 < 250:
            return "Very Poor"
        return "Severe"

    def _empty_probs(self, label: str) -> Dict[str, float]:
        probs = {name: 0.0 for name in self.CLASS_NAMES}
        probs[label] = 1.0
        return probs

    def train(self, X: np.ndarray, y: np.ndarray, min_rows: int = 200):
        """Train NN model and save best checkpoint. Falls back on small datasets."""
        n_rows = 0 if X is None else int(len(X))
        if n_rows < int(min_rows):
            print(f"Warning: training data < {int(min_rows)} rows. Using rule-based fallback.")
            return {
                "fallback": True,
                "reason": "insufficient_data",
                "rows": n_rows,
                "required_rows": int(min_rows),
            }

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print("Warning: need at least 2 classes to train NN. Using rule-based fallback.")
            return {
                "fallback": True,
                "reason": "insufficient_classes",
                "rows": n_rows,
                "class_count": int(len(unique_classes)),
            }

        _, class_counts = np.unique(y, return_counts=True)
        can_stratify = len(unique_classes) > 1 and n_rows >= 10 and int(class_counts.min()) >= 2

        tf_bundle = self._try_import_tf()
        if tf_bundle is None:
            # Train a practical fallback NN using sklearn so model training can proceed.
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y if can_stratify else None,
            )

            clf = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=0.001,
                max_iter=300,
                random_state=42,
            )
            clf.fit(X_train, y_train)
            joblib.dump(clf, self.sklearn_model_path)

            y_pred = clf.predict(X_test)
            acc = float((y_pred == y_test).mean())
            print(f"Sklearn NN fallback trained. Test accuracy: {acc:.4f}")
            print(
                classification_report(
                    y_test,
                    y_pred,
                    labels=list(range(len(self.CLASS_NAMES))),
                    target_names=self.CLASS_NAMES,
                    zero_division=0,
                )
            )

            cm = confusion_matrix(y_test, y_pred, labels=list(range(len(self.CLASS_NAMES))))
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xticks(range(len(self.CLASS_NAMES)), self.CLASS_NAMES, rotation=45, ha="right")
            plt.yticks(range(len(self.CLASS_NAMES)), self.CLASS_NAMES)
            plt.title("NN Confusion Matrix (sklearn fallback)")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig("outputs/nn_confusion_matrix.png", dpi=150)
            plt.close()

            return {
                "fallback": False,
                "backend": "sklearn_mlp",
                "test_accuracy": acc,
                "rows": n_rows,
            }

        (
            _tf,
            Sequential,
            Dense,
            Dropout,
            Adam,
            EarlyStopping,
            ModelCheckpoint,
            _load_model,
        ) = tf_bundle

        stratify_y = y if can_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify_y,
        )

        model = Sequential(
            [
                Dense(128, activation="relu", input_shape=(X.shape[1],)),
                Dropout(0.3),
                Dense(64, activation="relu"),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dense(6, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ModelCheckpoint(self.model_path, monitor="val_loss", save_best_only=True),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Final test accuracy: {test_acc:.4f}")

        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        print(
            classification_report(
                y_test,
                y_pred,
                labels=list(range(len(self.CLASS_NAMES))),
                target_names=self.CLASS_NAMES,
                zero_division=0,
            )
        )

        self._plot_training_curves(history.history, save_path="outputs/nn_training.png")
        self.evaluate(X_test, y_test)

        return {
            "fallback": False,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "history": history.history,
        }

    def _plot_training_curves(self, hist: Dict[str, list], save_path: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(hist.get("accuracy", []), label="train_acc")
        axes[0].plot(hist.get("val_accuracy", []), label="val_acc")
        axes[0].set_title("Accuracy")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(hist.get("loss", []), label="train_loss")
        axes[1].plot(hist.get("val_loss", []), label="val_loss")
        axes[1].set_title("Loss")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def predict_live(self, live_dict: Dict) -> Dict:
        """Predict live AQI class from current city readings."""
        feature_vals = []
        for col in self.FEATURE_ORDER:
            val = live_dict.get(col, 0.0)
            if val is None:
                val = 0.0
            feature_vals.append(float(val))

        x_raw = np.array(feature_vals, dtype=float).reshape(1, -1)

        scaler = None
        if os.path.exists(self.scaler_path):
            scaler = joblib.load(self.scaler_path)
            x_scaled = scaler.transform(x_raw)
        else:
            x_scaled = x_raw

        tf_bundle = self._try_import_tf()

        if os.path.exists(self.sklearn_model_path):
            clf = joblib.load(self.sklearn_model_path)
            probs_arr = clf.predict_proba(x_scaled)[0]
            all_probs = {name: 0.0 for name in self.CLASS_NAMES}
            for cls_idx, p in zip(clf.classes_, probs_arr):
                cls_int = int(cls_idx)
                if 0 <= cls_int < len(self.CLASS_NAMES):
                    all_probs[self.CLASS_NAMES[cls_int]] = float(p)

            idx = int(np.argmax([all_probs[name] for name in self.CLASS_NAMES]))
            label = self.CLASS_NAMES[idx]
            return {
                "predicted_category": label,
                "confidence": float(all_probs[label]),
                "all_probabilities": all_probs,
                "mode": "sklearn_mlp",
            }

        if tf_bundle is None or not os.path.exists(self.model_path):
            label = self._rule_based_label(float(live_dict.get("PM2.5", 0.0) or 0.0))
            return {
                "predicted_category": label,
                "confidence": 1.0,
                "all_probabilities": self._empty_probs(label),
                "mode": "fallback",
            }

        (
            _tf,
            _Sequential,
            _Dense,
            _Dropout,
            _Adam,
            _EarlyStopping,
            _ModelCheckpoint,
            load_model,
        ) = tf_bundle

        model = load_model(self.model_path)
        probs_arr = model.predict(x_scaled, verbose=0)[0]

        idx = int(np.argmax(probs_arr))
        label = self.CLASS_NAMES[idx]

        all_probs = {name: float(probs_arr[i]) for i, name in enumerate(self.CLASS_NAMES)}
        return {
            "predicted_category": label,
            "confidence": float(probs_arr[idx]),
            "all_probabilities": all_probs,
            "mode": "nn",
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Print confusion matrix and classification report."""
        tf_bundle = self._try_import_tf()
        if tf_bundle is None or not os.path.exists(self.model_path):
            print("Evaluation skipped: trained TensorFlow model not available.")
            return

        (
            _tf,
            _Sequential,
            _Dense,
            _Dropout,
            _Adam,
            _EarlyStopping,
            _ModelCheckpoint,
            load_model,
        ) = tf_bundle

        model = load_model(self.model_path)
        preds = np.argmax(model.predict(X_test, verbose=0), axis=1)

        cm = confusion_matrix(y_test, preds, labels=list(range(len(self.CLASS_NAMES))))

        try:
            import seaborn as sns  # noqa: PLC0415

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.CLASS_NAMES, yticklabels=self.CLASS_NAMES)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("NN Confusion Matrix")
            plt.tight_layout()
            plt.savefig("outputs/nn_confusion_matrix.png", dpi=150)
            plt.close()
        except Exception:
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xticks(range(len(self.CLASS_NAMES)), self.CLASS_NAMES, rotation=45, ha="right")
            plt.yticks(range(len(self.CLASS_NAMES)), self.CLASS_NAMES)
            plt.title("NN Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig("outputs/nn_confusion_matrix.png", dpi=150)
            plt.close()

        print(
            classification_report(
                y_test,
                preds,
                labels=list(range(len(self.CLASS_NAMES))),
                target_names=self.CLASS_NAMES,
                zero_division=0,
            )
        )
        print(f"Overall accuracy: {(preds == y_test).mean():.4f}")
