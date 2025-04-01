from itertools import product
from typing import Any, List, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class PartialDependence:
    """Partial Dependence"""

    def __init__(
        self,
        model: Any,
        X: pd.DataFrame,
        var_names: list[str],
        pred_type: Literal["regression", "classification"],
        label_list: list[str] = None,
    ) -> None:
        """
        Initializes the PartialDependence object.

        Args:
            model: Trained scikit-learn model.
            X: pandas DataFrame used for training.
            var_names: List of feature names.
            pred_type: Type of prediction ("regression" or "classification").
        """
        self.model = model
        self.X = X.copy()
        self.var_names = var_names
        self.pred_type = pred_type
        self.label_list = label_list

    def partial_dependence(self, var_name: str, n_grid: int = 50) -> pd.DataFrame:
        """
        Calculates the partial dependence for given variables.

        Args:
            var_name: Variable name.
            n_grid: Number of grid points per variable.

        Returns:
            A pandas DataFrame containing the grid points and average predictions.
        """

        if not isinstance(self.X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        grid_ranges = {}
        grid_ranges[var_name] = np.linspace(
            self.X[var_name].min(), self.X[var_name].max(), num=n_grid
        )

        grid_points = list(product(*grid_ranges.values()))
        grid_points_df = pd.DataFrame(grid_points, columns=[var_name])
        if self.pred_type == "regression":
            average_predictions = []
            for _, row in grid_points_df.iterrows():
                average_predictions.append(self._counterfactual_prediction(row).mean())
            grid_points_df["avg_pred"] = average_predictions
        else:
            proba_predictions = []
            for _, row in grid_points_df.iterrows():
                proba_predictions.append(
                    self._counterfactual_prediction(row).mean(axis=0)
                )
            grid_points_df[self.label_list] = proba_predictions
        return grid_points_df

    def _counterfactual_prediction(
        self, grid_point_series: dict[str:float]
    ) -> np.ndarray:
        """Makes counterfactual predictions."""

        X_counterfactual = self.X.copy()
        for (
            var_name,
            grid_point,
        ) in grid_point_series.items():
            X_counterfactual[var_name] = grid_point

        if self.pred_type == "regression":
            return self.model.predict(X_counterfactual)
        else:
            return self.model.predict_proba(X_counterfactual)


class IndividualConditionalExpectation(PartialDependence):
    """Individual Conditional Expectation"""

    def individual_conditional_expectation(
        self, var_name: str, n_grid: int = 50
    ) -> None:
        """Calculates Individual Conditional Expectation (ICE) values.

        Args:
            var_name: The name of the feature for which to calculate ICE.
            n_grid: The number of grid points to use for evaluating the feature's range.  A finer grid (larger value) can capture more nuanced relationships but may be more computationally expensive and potentially noisy.  A coarser grid may miss important details.  Defaults to 50.
        """
        ids_to_compute = [i for i in range(self.X.shape[0])]
        self.target_var_name = var_name
        value_range = np.linspace(
            self.X[var_name].min(), self.X[var_name].max(), num=n_grid
        )
        if self.pred_type == "regression":
            individual_prediction = np.array(
                [
                    self._counterfactual_prediction({self.target_var_name: x})[
                        ids_to_compute
                    ]
                    for x in value_range
                ]
            )
            self.df_ice = (
                pd.DataFrame(data=individual_prediction, columns=ids_to_compute)
                .assign(**{var_name: value_range})
                .melt(id_vars=var_name, var_name="instance", value_name="ice")
            )
        else:
            individual_prediction = np.array(
                [
                    self._counterfactual_prediction({self.target_var_name: x})
                    for x in value_range
                ]
            )
            self.individual_prediction = individual_prediction
            ice_list = []
            for i in range(individual_prediction.shape[-1]):
                df_ice_one_category = pd.DataFrame(
                    individual_prediction[:, :, i], columns=ids_to_compute
                )
                df_ice_one_category = df_ice_one_category.assign(
                    **{var_name: value_range}
                ).melt(id_vars=var_name, var_name="instance", value_name="ice")
                df_ice_one_category["category"] = i
                ice_list.append(df_ice_one_category)
            self.df_ice = pd.concat(ice_list, ignore_index=True, axis=0)

        self.ids_to_compute = ids_to_compute
        self.individual_prediction = individual_prediction

        min_ice_by_instance = self.df_ice.loc[
            self.df_ice.groupby("instance")[var_name].idxmin()
        ]
        min_ice_dict = dict(
            zip(
                min_ice_by_instance["instance"],
                min_ice_by_instance["ice"],
                strict=False,
            )
        )
        self.df_ice["ice_diff"] = self.df_ice.apply(
            lambda row: row["ice"] - min_ice_dict[row["instance"]], axis=1
        )

    def plot_ice(
        self,
        fig: Any = None,
        ax: Any = None,
        ylim: Union[List[float], None] = None,
    ) -> None:
        """Visualizes ICE plots.

        Args:
            ylim: The range of the y-axis. If not specified, the range of the ICE curves will be used.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        if "category" not in self.df_ice.columns:
            sns.lineplot(
                x=self.target_var_name,
                y="ice",
                units="instance",
                data=self.df_ice,
                lw=0.8,
                alpha=0.3,
                estimator=None,
                zorder=1,
                ax=ax,
                color="black",
            )
            ax.set(xlabel=self.target_var_name, ylabel="Prediction", ylim=ylim)
            ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            fig.suptitle(f"Individual Conditional Expectation({self.target_var_name})")
        else:
            categories = self.df_ice["category"].unique()
            num_categories = len(categories)
            cmap = plt.get_cmap("viridis", num_categories)
            for i, category in enumerate(categories):
                df_subset = self.df_ice.query("category == @category")
                sns.lineplot(
                    x=self.target_var_name,
                    y="ice",
                    units="instance",
                    data=df_subset,
                    lw=0.8,
                    alpha=0.3,
                    estimator=None,
                    zorder=1,
                    ax=ax,
                    color=cmap(i),
                    label=category,
                )
            ax.set(xlabel=self.target_var_name, ylabel="Prediction", ylim=ylim)
            ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            fig.suptitle(f"Individual Conditional Expectation({self.target_var_name})")

    def plot_ice_with_average(
        self, fig: Any = None, ax: Any = None, ylim: Union[List[float], None] = None
    ) -> None:
        """Visualizes ICE (Individual Conditional Expectation) plots.

        Args:
            ylim: Tuple specifying the y-axis limits (min, max). If None, the limits are automatically determined based on the range of the ICE values.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        sns.lineplot(
            x=self.target_var_name,
            y="ice_diff",
            units="instance",
            data=self.df_ice,
            lw=0.8,
            alpha=0.5,
            estimator=None,
            zorder=1,
            color="black",
            ax=ax,
        )
        average_ice = (
            self.df_ice.groupby(self.target_var_name)["ice_diff"].mean().reset_index()
        )
        sns.lineplot(
            x=self.target_var_name,
            y="ice_diff",
            data=average_ice,
            color="yellow",
            lw=5,
            ax=ax,
        )
        ax.set(xlabel=self.target_var_name, ylabel="Prediction", ylim=ylim)
        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        fig.suptitle(f"Individual Conditional Expectation({self.target_var_name})")
