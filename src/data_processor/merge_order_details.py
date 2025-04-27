import math
from typing import Optional

import pandas as pd


def _create_payment_agg_data(
    payments_dataset_path: str,
) -> pd.DataFrame:
    """
    Aggregates payment data by order ID.

    Args:
        payments_dataset_path (str): Path to the payments dataset CSV file.

    Returns:
        pd.DataFrame: Aggregated payment data with columns like total payment value,
                    payment type count, and most common payment type.
    """
    df_payments = pd.read_csv(payments_dataset_path)
    agg_payments = (
        df_payments.groupby("order_id")
        .agg(
            {
                "payment_value": "sum",
                "payment_type": [
                    "nunique",
                    lambda x: x.value_counts().index[0],
                ],
            }
        )
        .reset_index()
    )
    agg_payments.columns = [
        "order_id",
        "total_payment_value",
        "payment_type_count",
        "most_common_payment_type",
    ]

    payment_type_by_value = (
        df_payments.groupby(["order_id", "payment_type"])["payment_value"]
        .sum()
        .reset_index()
    )
    max_payment_type = (
        payment_type_by_value.sort_values("payment_value", ascending=False)
        .groupby("order_id")
        .first()[["payment_type", "payment_value"]]
        .reset_index()
        .rename(
            columns={
                "payment_type": "highest_value_payment_type",
                "payment_value": "highest_payment_value",
            }
        )
    )
    df_agg_payments = pd.merge(
        agg_payments, max_payment_type, on="order_id", how="left"
    )
    assert df_agg_payments.shape[0] == agg_payments.shape[0]
    return df_agg_payments


def _create_geolocation_agg_data(
    geolocation_dataset_path: str,
) -> pd.DataFrame:
    """
    Aggregates geolocation data by state and zip code prefix.

    Args:
        geolocation_dataset_path (str): Path to the geolocation dataset CSV file.
    Returns:
        pd.DataFrame: Aggregated geolocation data with mean latitude and longitude for each state and zip code prefix.
    """
    df_geolocation = pd.read_csv(geolocation_dataset_path)
    df_geolocation_agg = (
        df_geolocation.groupby(["geolocation_state", "geolocation_zip_code_prefix"])
        .agg(
            {
                "geolocation_lat": "mean",
                "geolocation_lng": "mean",
            }
        )
        .reset_index()
    )
    return df_geolocation_agg


def _merge_order_payment_geolocation_data(
    order_items_dataset_path: str,
    geolocation_dataset_path: str,
    payments_dataset_path: str,
) -> pd.DataFrame:
    """
    Merges order items, payment, and geolocation data.

    Args:
        order_items_dataset_path (str): Path to the order items dataset CSV file.
        geolocation_dataset_path (str): Path to the geolocation dataset CSV file.
        payments_dataset_path (str): Path to the payments dataset CSV file.

    Returns:
        pd.DataFrame: Merged DataFrame containing order items, payment, and customer geolocation data.
    """
    df_agg_payments = _create_payment_agg_data(payments_dataset_path)
    df_geolocation_agg = _create_geolocation_agg_data(geolocation_dataset_path)

    df_order_items = pd.read_csv(order_items_dataset_path)
    df_order_payment_merged = pd.merge(
        df_order_items, df_agg_payments, on="order_id", how="left"
    )
    assert df_order_payment_merged.shape[0] == df_order_items.shape[0]
    df_order_payment_geolocation_merged = pd.merge(
        df_order_payment_merged,
        df_geolocation_agg.rename(
            columns={
                "geolocation_lat": "customer_lat",
                "geolocation_lng": "customer_lng",
            }
        ),
        left_on=["customer_zip_code_prefix", "customer_state"],
        right_on=["geolocation_zip_code_prefix", "geolocation_state"],
        how="left",
    )
    assert (
        df_order_payment_geolocation_merged.shape[0] == df_order_payment_merged.shape[0]
    )
    return df_order_payment_geolocation_merged


def _create_seller_info_data(
    item_product_seller_merged_path: str,
    geolocation_dataset_path: str,
) -> pd.DataFrame:
    """
    Creates aggregated seller information by order ID.

    Args:
        item_product_seller_merged_path (str): Path to the merged item, product, and seller dataset CSV file.
        geolocation_dataset_path (str): Path to the geolocation dataset CSV file.

    Returns:
        pd.DataFrame: Aggregated seller information including mean latitude, longitude, price,
                    number of unique products, and other relevant seller-related features.
    """
    df_item_product_seller_merged = pd.read_csv(item_product_seller_merged_path)
    df_geolocation_agg = _create_geolocation_agg_data(geolocation_dataset_path)
    df_item_products_seller_geolocation_merged = pd.merge(
        df_item_product_seller_merged,
        df_geolocation_agg.rename(
            columns={"geolocation_lat": "seller_lat", "geolocation_lng": "seller_lng"}
        ),
        left_on=["seller_zip_code_prefix", "seller_state"],
        right_on=["geolocation_zip_code_prefix", "geolocation_state"],
        how="left",
    )
    assert (
        df_item_products_seller_geolocation_merged.shape[0]
        == df_item_product_seller_merged.shape[0]
    )
    df_item_products_seller_geolocation_merged["shipping_limit_date"] = pd.to_datetime(
        df_item_products_seller_geolocation_merged["shipping_limit_date"]
    )
    df_item_products_seller_geolocation_merged["product_volume_cm3"] = (
        df_item_products_seller_geolocation_merged["product_length_cm"]
        * df_item_products_seller_geolocation_merged["product_height_cm"]
        * df_item_products_seller_geolocation_merged["product_width_cm"]
    )
    seller_info_agg_by_order_id = (
        df_item_products_seller_geolocation_merged.groupby("order_id")
        .agg(
            {
                "shipping_limit_date": "mean",
                "price": "sum",
                "freight_value": "sum",
                "sum_price_freight_by_order": "sum",
                "seller_lat": "mean",
                "seller_lng": "mean",
                "product_id": "nunique",
                "seller_id": "nunique",
                "product_photos_qty": "mean",
                "product_weight_g": "sum",
                "product_length_cm": "mean",
                "product_height_cm": "mean",
                "product_width_cm": "mean",
                "product_volume_cm3": "sum",
                "seller_city": "nunique",
                "seller_state": "nunique",
            }
        )
        .rename(
            columns={
                "price": "sum_price",
                "freight_value": "sum_freight",
                "product_id": "product_count",
                "product_volume_cm3": "sum_product_volume_cm3",
                "product_weight_g": "sum_product_weight_g",
                "seller_id": "seller_count",
                "seller_city": "seller_city_count",
                "seller_state": "seller_state_count",
            }
        )
    ).reset_index()

    return seller_info_agg_by_order_id


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculates the great-circle distance between two points on Earth using the Haversine formula.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        lng1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lng2 (float): Longitude of the second point in degrees.

    Returns:
        float: The distance between the two points in kilometers.
    """
    R = 6371
    lat1_rad = math.radians(lat1)
    lng1_rad = math.radians(lng1)
    lat2_rad = math.radians(lat2)
    lng2_rad = math.radians(lng2)

    dlng = lng2_rad - lng1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def _create_distance_info_between_customer_and_seller(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the distance between customer and seller locations and creates distance bins.

    Args:
        df (pd.DataFrame): DataFrame containing customer and seller latitude and longitude data.

    Returns:
        pd.DataFrame: DataFrame with added 'distance_between_customer_and_seller' column
                    and one-hot encoded distance bin columns.
    """
    df["distance_between_customer_and_seller"] = df.apply(
        lambda row: _haversine(
            row["customer_lat"],
            row["customer_lng"],
            row["seller_lat"],
            row["seller_lng"],
        ),
        axis=1,
    )
    df["distance_bin"] = pd.qcut(
        df["distance_between_customer_and_seller"],
        q=5,
        labels=[f"Bin {i + 1}" for i in range(5)],
    )
    distance_bin_dummies = pd.get_dummies(
        df["distance_bin"], prefix="distance_bin"
    ).astype(int)
    df_merged = pd.concat([df, distance_bin_dummies], axis=1)
    assert df_merged.shape[0] == df.shape[0]
    return df_merged


def _crete_span_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time span columns based on order timestamps.

    Args:
        df (pd.DataFrame): DataFrame containing order timestamps.

    Returns:
        pd.DataFrame: DataFrame with added columns representing the time span in hours
                    between various order events.
    """
    df_copy = df.copy()
    for col in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "shipping_limit_date",
    ]:
        df_copy[col] = pd.to_datetime(df_copy[col])
    df_copy["order_approved_span_hours_from_purchase"] = pd.to_numeric(
        (
            df_copy["order_approved_at"] - df_copy["order_purchase_timestamp"]
        ).dt.total_seconds()
        / 3600,
        errors="coerce",
    )
    df_copy["order_shipping_limit_span_hours_from_purchase"] = pd.to_numeric(
        (
            df_copy["shipping_limit_date"] - df_copy["order_purchase_timestamp"]
        ).dt.total_seconds()
        / 3600,
        errors="coerce",
    )
    df_copy["order_estimated_delivery_span_hours_from_carrier"] = pd.to_numeric(
        (
            df_copy["order_estimated_delivery_date"] - df_copy["shipping_limit_date"]
        ).dt.total_seconds()
        / 3600,
        errors="coerce",
    )
    df_copy["order_estimated_delivery_span_hours_from_purchase"] = pd.to_numeric(
        (
            df_copy["order_estimated_delivery_date"]
            - df_copy["order_purchase_timestamp"]
        ).dt.total_seconds()
        / 3600,
        errors="coerce",
    )
    df_copy["order_estimated_delivery_span_hours_from_shipping_limit"] = pd.to_numeric(
        (
            df_copy["order_estimated_delivery_date"] - df_copy["shipping_limit_date"]
        ).dt.total_seconds()
        / 3600,
        errors="coerce",
    )
    df_copy["order_delivered_carrier_span_hours_from_purchase"] = pd.to_numeric(
        (
            df_copy["order_delivered_carrier_date"]
            - df_copy["order_purchase_timestamp"]
        ).dt.total_seconds()
        / 3600,
        errors="coerce",
    )
    df_copy["order_delivered_carrier_span_hours_from_approval"] = pd.to_numeric(
        (
            df_copy["order_delivered_carrier_date"] - df_copy["order_approved_at"]
        ).dt.total_seconds()
        / 3600,
        errors="coerce",
    )
    df_copy["order_shipping_limit_span_hours_from_approval"] = pd.to_numeric(
        (
            df_copy["shipping_limit_date"] - df_copy["order_approved_at"]
        ).dt.total_seconds()
        / 3600,
        errors="coerce",
    )
    df_copy["order_delivered_customer_span_hours_from_purchase"] = pd.to_numeric(
        (
            df_copy["order_delivered_customer_date"]
            - df_copy["order_purchase_timestamp"]
        ).dt.total_seconds()
        / 3600,
        errors="coerce",
    )
    df_copy["order_delivered_customer_span_hours_from_carrier"] = pd.to_numeric(
        (
            df_copy["order_delivered_customer_date"]
            - df_copy["order_delivered_carrier_date"]
        ).dt.total_seconds()
        / 3600,
        errors="coerce",
    )
    df_copy["order_delivered_carrier_span_hours_from_limit_date"] = pd.to_numeric(
        (
            df_copy["order_delivered_carrier_date"] - df_copy["shipping_limit_date"]
        ).dt.total_seconds()
        / 3600,
        errors="coerce",
    )
    df_copy["order_delivered_customer_span_hours_from_limit_date"] = pd.to_numeric(
        (
            df_copy["order_delivered_customer_date"]
            - df_copy["order_estimated_delivery_date"]
        ).dt.total_seconds()
        / 3600,
    )
    return df_copy


def _get_most_common_category(x: pd.Series) -> str:
    """
    Gets the most common category from a Pandas Series.

    Args:
        x (pd.Series): A Pandas Series containing categories.

    Returns:
        str: The most frequent category in the Series. Returns None if the Series is empty.
    """
    value_counts = x.value_counts()
    return value_counts.index[0] if len(value_counts) > 0 else None


def _merge_most_common_category_by_order_id(
    df: pd.DataFrame, item_product_seller_merged_path: str
) -> pd.DataFrame:
    """
    Merges the most common product category, seller city, and seller state with the main DataFrame.

    Args:
        df (pd.DataFrame): The main DataFrame to merge into.
        item_product_seller_merged_path (str): Path to the item_product_seller_merged dataset CSV file.

    Returns:
        pd.DataFrame: The merged DataFrame containing the most common product category, seller city, and seller state for each order ID.
    """
    df_copy = df.copy()
    df_item_product_seller_merged = pd.read_csv(item_product_seller_merged_path)
    agg_most_common_category_by_order_id = (
        df_item_product_seller_merged.groupby("order_id")[
            ["product_category_name_english", "seller_city", "seller_state"]
        ]
        .agg(
            _get_most_common_category,
        )
        .reset_index()
    )
    df_merged = pd.merge(
        df_copy,
        agg_most_common_category_by_order_id,
        on="order_id",
        how="left",
    )
    assert df_merged.shape[0] == df_copy.shape[0]
    return df_merged


def process_order_details_data(
    order_items_dataset_path: str,
    geolocation_dataset_path: str,
    payments_dataset_path: str,
    item_product_seller_merged_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Processes and merges order details data from multiple datasets.

    Args:
        order_items_dataset_path (str): Path to the order items dataset CSV file.
        geolocation_dataset_path (str): Path to the geolocation dataset CSV file.
        payments_dataset_path (str): Path to the payments dataset CSV file.
        item_product_seller_merged_path (str): Path to the merged item, product, and seller dataset CSV file.
        output_path (Optional[str]): Optional path to save the processed DataFrame as a CSV file.

    Returns:
        pd.DataFrame: A merged DataFrame containing order details, payment information,
                    geolocation data, seller information, distance data, and time span data.
    """
    # 1. Merge payment information and geolocation information by order ID
    df_order_payment_geolocation_merged = _merge_order_payment_geolocation_data(
        order_items_dataset_path=order_items_dataset_path,
        geolocation_dataset_path=geolocation_dataset_path,
        payments_dataset_path=payments_dataset_path,
    )
    # 2. Merge seller information by order ID
    seller_info_agg_by_order_id = _create_seller_info_data(
        item_product_seller_merged_path=item_product_seller_merged_path,
        geolocation_dataset_path=geolocation_dataset_path,
    )
    # 3. Merge seller information by store ID
    df_order_payment_geolocation_seller_merged = pd.merge(
        df_order_payment_geolocation_merged,
        seller_info_agg_by_order_id,
        on="order_id",
        how="left",
    )
    assert (
        df_order_payment_geolocation_seller_merged.shape[0]
        == df_order_payment_geolocation_merged.shape[0]
    )
    # 4. Merge the most frequent category by order ID
    df_order_payment_geolocation_seller_merged = (
        _merge_most_common_category_by_order_id(
            df=df_order_payment_geolocation_seller_merged,
            item_product_seller_merged_path=item_product_seller_merged_path,
        )
    )
    # 5. Merge period/time span information by order ID
    df_order_payment_geolocation_seller_merged = _crete_span_cols(
        df=df_order_payment_geolocation_seller_merged
    )
    # 6. Merge distance information between customer and seller by order ID
    df_order_payment_geolocation_seller_merged = (
        _create_distance_info_between_customer_and_seller(
            df=df_order_payment_geolocation_seller_merged
        )
    )
    if output_path is not None:
        df_order_payment_geolocation_seller_merged.to_csv(output_path, index=False)
    return df_order_payment_geolocation_seller_merged
