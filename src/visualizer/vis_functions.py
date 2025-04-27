from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_monthly_trend(
    df,
    agg_type: Literal["n_sales", "sales", "user", "seller"] = "n_sales",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plots the monthly trend for various aggregation types (number of sales, total sales, number of users, number of sellers).

    Args:
        df (pd.DataFrame): The input DataFrame containing order data with 'order_purchase_timestamp' column.
        agg_type (Literal["n_sales", "sales", "user", "seller"], optional): The type of aggregation to plot.
            - "n_sales": Number of orders.
            - "sales": Total sales amount.
            - "user": Number of unique users.
            - "seller": Number of unique sellers.
            Defaults to "n_sales".
        output_path (Optional[Path], optional): Path to save the plot as an image file. Defaults to None.

    Returns:
        None: Displays the plot using matplotlib.
    """
    date_range = pd.period_range(start="2016-09", end="2018-10", freq="M")
    if agg_type == "n_sales":
        monthly_agg = (
            df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))["order_id"]
            .count()
            .reindex(date_range, fill_value=0)
            .reset_index(name="y")
        )
        y_label = "Number of Orders"
        y_max = 8000
    elif agg_type == "sales":
        monthly_agg = (
            df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))["price"]
            .sum()
            .reindex(date_range, fill_value=0)
            .reset_index(name="y")
        )
        y_label = "Total Sales (R$)"
        y_max = 1100000
    elif agg_type == "user":
        monthly_agg = (
            df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))[
                "customer_unique_id"
            ]
            .nunique()
            .reindex(date_range, fill_value=0)
            .reset_index(name="y")
        )
        y_label = "Number of Users"
        y_max = 8000
    else:
        monthly_agg = (
            df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))["seller_id"]
            .nunique()
            .reindex(date_range, fill_value=0)
            .reset_index(name="y")
        )
        y_label = "Number of Sellers"
        y_max = 1700

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(
        monthly_agg["index"].astype(str),
        monthly_agg["y"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="blue",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Month", fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.tick_params(axis="x", rotation=45, labelsize=16)
    ax.tick_params(axis="y", labelsize=16)

    x_labels = monthly_agg["index"].astype(str)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(
        [label if i % 2 == 0 else "" for i, label in enumerate(x_labels)], rotation=45
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax.set_ylim(0, y_max)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    plt.show()


def plot_category_sales_trend(
    categories: list,
    df_merged: pd.DataFrame,
    figsize: tuple = (12, 6),
    title: str = None,
    start_date: str = None,
    end_date: str = None,
    agg_type: Literal["n_sales", "sales", "user"] = "n_sales",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plots the monthly sales trend for specified product categories.

    Args:
        categories (list): A list of product category names to plot.
        df_merged (pd.DataFrame): The input DataFrame containing order and product information.
            Must have 'order_purchase_timestamp', 'order_purchase_month',
            'product_category_name_english', 'order_id' (for 'n_sales'), and 'price' (for 'sales') columns.
        figsize (tuple, optional): The figure size for the plot. Defaults to (12, 6).
        title (str, optional): The title of the plot. Defaults to None.
        start_date (str, optional): The start date for the plot (YYYY-MM-DD). If None, the earliest date in the data is used. Defaults to None.
        end_date (str, optional): The end date for the plot (YYYY-MM-DD). If None, the latest date in the data is used. Defaults to None.
        agg_type (Literal["n_sales", "sales", "user"], optional): The type of aggregation to plot.
            - "n_sales": Number of orders.
            - "sales": Total sales amount.
            - "user": Number of unique users.
            Defaults to "n_sales".
        output_path (Optional[Path], optional): Path to save the plot as an image file. Defaults to None.

    Returns:
        None: Displays the plot using matplotlib.  Fills missing months with 0 to ensure continuous trend lines.
    """

    df_merged["order_purchase_timestamp"] = pd.to_datetime(
        df_merged["order_purchase_timestamp"]
    )
    df_merged["order_purchase_month"] = pd.to_datetime(
        df_merged["order_purchase_month"]
    )
    if start_date is None:
        start_date = df_merged["order_purchase_timestamp"].min()

    if end_date is None:
        end_date = df_merged["order_purchase_timestamp"].max()

    all_months = df_merged.query(
        "order_purchase_month >= @start_date & order_purchase_month <= @end_date"
    )["order_purchase_month"].unique()

    all_categories = df_merged["product_category_name_english"].unique()

    month_category_combinations = pd.MultiIndex.from_product(
        [all_months, all_categories],
        names=["order_purchase_month", "product_category_name_english"],
    )

    if agg_type == "n_sales":
        monthly_category_counts = (
            df_merged.groupby(
                ["order_purchase_month", "product_category_name_english"]
            )["order_id"]
            .nunique()
            .reindex(month_category_combinations, fill_value=0)
            .reset_index(name="sum")
            .sort_values("order_purchase_month")
        )
    elif agg_type == "sales":
        monthly_category_counts = (
            df_merged.groupby(
                ["order_purchase_month", "product_category_name_english"]
            )["price"]
            .sum()
            .reindex(month_category_combinations, fill_value=0)
            .reset_index(name="sum")
            .sort_values("order_purchase_month")
        )
    else:
        monthly_category_counts = (
            df_merged.groupby(
                ["order_purchase_month", "product_category_name_english"]
            )["customer_unique_id"]
            .nunique()
            .reindex(month_category_combinations, fill_value=0)
            .reset_index(name="sum")
            .sort_values("order_purchase_month")
        )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i, category in enumerate(categories):
        category_data = monthly_category_counts.query(
            "product_category_name_english == @category"
        )
        if i < 5:
            ax.plot(
                category_data["order_purchase_month"].astype(str),
                category_data["sum"],
                marker="o",
                linewidth=2,
                label=category,
            )
        elif i >= 10:
            ax.plot(
                category_data["order_purchase_month"].astype(str),
                category_data["sum"],
                marker="o",
                linewidth=2,
                label=category,
                alpha=0.3,
                color="black",
            )
        else:
            ax.plot(
                category_data["order_purchase_month"].astype(str),
                category_data["sum"],
                marker="o",
                linewidth=2,
                label=category,
                alpha=0.6,
            )
    if title is None:
        title = "Monthly Sales Volume by Categories"

    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("Total Sales (R$)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.0, 1), loc="upper left", fontsize=12)
    x_labels = category_data["order_purchase_month"].astype(str)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(
        [label if i % 2 == 1 else "" for i, label in enumerate(x_labels)], rotation=45
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    plt.show()
