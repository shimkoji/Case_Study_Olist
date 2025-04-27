from typing import Optional

import pandas as pd


def process_order_customer_data(
    customer_dataset_path: str,
    orders_dataset_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads, processes, and merges order and customer datasets, adding various date-related features.

    Args:
        customer_dataset_path (str): Path to the CSV file containing customer data.
        orders_dataset_path (str): Path to the CSV file containing order data.
        output_path (Optional[str], optional): If provided, the merged DataFrame will be saved as a CSV to this path.

    Returns:
        pd.DataFrame: The merged DataFrame containing both order and customer information, with additional date and delivery features.

    Function Details:
        - Reads customer and order data from the specified CSV files.
        - Converts relevant order date columns to datetime.
        - Extracts and adds new columns for month, date, and weekday for various order-related timestamps.
        - Adds a boolean column 'is_delivery_to_customers_delayed' indicating if the actual delivery date was later than the estimated delivery date.
        - Merges the order and customer DataFrames on 'customer_id' using a left join.
        - Optionally saves the merged DataFrame to a CSV file if 'output_path' is specified.
        - Returns the merged DataFrame for further analysis or processing.
    """
    df_customer = pd.read_csv(customer_dataset_path)
    df_orders = pd.read_csv(orders_dataset_path)
    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_columns:
        df_orders[col] = pd.to_datetime(df_orders[col])

    df_orders["order_purchase_month"] = df_orders[
        "order_purchase_timestamp"
    ].dt.to_period("M")
    df_orders["order_purchase_date"] = df_orders[
        "order_purchase_timestamp"
    ].dt.to_period("D")
    df_orders["order_purchase_weekday"] = df_orders[
        "order_purchase_timestamp"
    ].dt.day_name()

    df_orders["order_approved_month"] = df_orders["order_approved_at"].dt.to_period("M")
    df_orders["order_approved_date"] = df_orders["order_approved_at"].dt.to_period("D")
    df_orders["order_approved_weekday"] = df_orders["order_approved_at"].dt.day_name()
    df_orders["order_delivered_carrier_month"] = df_orders[
        "order_delivered_carrier_date"
    ].dt.to_period("M")
    df_orders["order_delivered_carrier_weekday"] = df_orders[
        "order_delivered_carrier_date"
    ].dt.day_name()
    df_orders["order_delivered_customer_month"] = df_orders[
        "order_delivered_customer_date"
    ].dt.to_period("M")
    df_orders["order_delivered_customer_weekday"] = df_orders[
        "order_delivered_customer_date"
    ].dt.day_name()
    df_orders["order_estimated_delivery_month"] = df_orders[
        "order_estimated_delivery_date"
    ].dt.to_period("M")
    df_orders["order_estimated_delivery_weekday"] = df_orders[
        "order_estimated_delivery_date"
    ].dt.day_name()

    df_orders["is_delivery_to_customers_delayed"] = (
        df_orders["order_delivered_customer_date"]
        > df_orders["order_estimated_delivery_date"]
    )
    df_orders_customer_merged = pd.merge(
        df_orders, df_customer, on="customer_id", how="left"
    )
    assert df_orders.shape[0] == df_orders_customer_merged.shape[0]

    if output_path is not None:
        df_orders_customer_merged.to_csv(output_path, index=False)
    return df_orders_customer_merged
