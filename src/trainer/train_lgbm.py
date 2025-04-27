import logging
import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class LGBMTrainer:
    def __init__(
        self, x: pd.DataFrame, y: pd.Series, params: dict, categorical_features: list
    ):
        self.x = x
        self.y = y
        self.params = params
        self.categorical_features = categorical_features
        self.seed = 42

    def train(self, best_model_output_path: Optional[Path] = None):
        for col in self.categorical_features:
            if col in self.x.columns:
                self.x[col] = self.x[col].astype("category")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=self.seed
        )
        lgb_reg = LGBMRegressor(
            objective="regression",
            metric="rmse",
            boosting_type="gbdt",
            verbose=-1,
            random_state=self.seed,
        )
        random_search = RandomizedSearchCV(
            estimator=lgb_reg,
            param_distributions=self.params,
            n_iter=50,
            scoring="neg_root_mean_squared_error",
            cv=5,
            random_state=self.seed,
            n_jobs=-1,
            verbose=2,
        )
        random_search.fit(
            self.X_train, self.y_train, categorical_feature=self.categorical_features
        )
        logger.info(f"Best params: {random_search.best_params_}")
        logger.info(
            "Best cross-validation RMSE: {:.4f}".format(-random_search.best_score_)
        )
        self.best_model = random_search.best_estimator_
        self.y_pred = self.best_model.predict(self.X_test)
        if best_model_output_path is not None:
            with open(best_model_output_path, "wb") as f:
                pickle.dump(self.best_model, f)

    def plot_actual_vs_predicted(self, output_path: Optional[Path] = None):
        plt.figure(figsize=(7, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.5)
        plt.plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            "r--",
            lw=2,
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values (Best Model)")
        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path)
        plt.show()

    def plot_feature_importance(self, output_path: Optional[Path] = None):
        df_importance = pd.DataFrame(
            {
                "feature": self.X_train.columns,
                "importance": self.best_model.feature_importances_,
            }
        )
        df_importance = df_importance.sort_values("importance", ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=df_importance.head(20), x="importance", y="feature", color="blue"
        )
        plt.xlabel("Importance", fontsize=16)
        plt.ylabel("")
        plt.tick_params(axis="x", labelsize=14)
        plt.tick_params(axis="y", labelsize=18)
        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path)
        plt.show()

    def plot_shap_values(self, output_path: Optional[Path] = None):
        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(self.X_train)
        shap_explanation = shap.Explanation(
            values=shap_values,
            feature_names=self.X_train.columns,
            data=self.X_train.values,
        )

        shap.initjs()
        shap.summary_plot(
            shap_values,
            self.X_train,
            plot_type="bar",
            plot_size=[12, 8],
            show=False,
        )
        plt.tick_params(axis="y", labelsize=18)
        plt.tick_params(axis="x", labelsize=14)
        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path)
        plt.show()

    def plot_ice_curves(self, output_path: Optional[Path] = None):
        plt.figure(figsize=(12, 6))
        feature_values = np.sort(
            self.X_train["distance_between_customer_and_seller"].unique()
        )
        ice_values = []
        for i in range(min(100, len(self.X_train))):
            sample = self.X_train.iloc[[i]].copy()
            ice_sample = []

            for value in feature_values:
                temp_sample = sample.copy()
                temp_sample["distance_between_customer_and_seller"] = value
                pred = self.best_model.predict(temp_sample)
                ice_sample.append(pred[0])

            ice_values.append(ice_sample)

        ice_values = np.array(ice_values)
        mean_ice = np.mean(ice_values, axis=0)

        plt.plot(feature_values, mean_ice, "b-", linewidth=2, label="Average ICE")
        for i in range(min(10, len(ice_values))):
            plt.plot(feature_values, ice_values[i], "gray", alpha=0.3)

        plt.xlabel("Distance between Customer and Seller")
        plt.ylabel("Predicted Delivery Time")
        plt.title("ICE Plot for Distance between Customer and Seller")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path)
        plt.show()
