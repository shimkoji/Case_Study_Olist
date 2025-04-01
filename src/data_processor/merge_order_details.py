import math
from typing import Optional

import pandas as pd


def _create_payment_agg_data(
    payments_dataset_path: str,
) -> pd.DataFrame:
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


def _haversine(lat1, lng1, lat2, lng2):
    """
    2点間の大圏距離を計算する（Haversine formulaを使用）

    Args:
        lat1: 1点目の緯度 (度)
        lng1: 1点目の経度 (度)
        lat2: 2点目の緯度 (度)
        lng2: 2点目の経度 (度)

    Returns:
        2点間の距離 (km)
    """
    R = 6371  # 地球の半径 (km)

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

    # 元のデータフレームに結合
    df_merged = pd.concat([df, distance_bin_dummies], axis=1)
    assert df_merged.shape[0] == df.shape[0]
    return df_merged


def _crete_span_cols(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for col in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        df_copy[col] = pd.to_datetime(df_copy[col])
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
    df_copy["order_delivered_carrier_span_hours_from_purchase"] = pd.to_numeric(
        (
            df_copy["order_delivered_carrier_date"]
            - df_copy["order_purchase_timestamp"]
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


def _get_most_common_category(x):
    value_counts = x.value_counts()
    return value_counts.index[0] if len(value_counts) > 0 else None


def _merge_most_common_category_by_order_id(
    df: pd.DataFrame, item_product_seller_merged_path: str
) -> pd.DataFrame:
    df_copy = df.copy()
    df_item_product_seller_merged = pd.read_csv(item_product_seller_merged_path)
    # 最も多いカテゴリを取得
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
    # 1. 注文IDごとに支払い情報とジオロケーション情報を結合
    df_order_payment_geolocation_merged = _merge_order_payment_geolocation_data(
        order_items_dataset_path=order_items_dataset_path,
        geolocation_dataset_path=geolocation_dataset_path,
        payments_dataset_path=payments_dataset_path,
    )
    # 2. 注文IDごとにセラー情報を結合
    seller_info_agg_by_order_id = _create_seller_info_data(
        item_product_seller_merged_path=item_product_seller_merged_path,
        geolocation_dataset_path=geolocation_dataset_path,
    )
    # 3. 注文IDごとに販売店情報を結合
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
    # 4. 注文IDごとに最も多いカテゴリを結合
    df_order_payment_geolocation_seller_merged = (
        _merge_most_common_category_by_order_id(
            df=df_order_payment_geolocation_seller_merged,
            item_product_seller_merged_path=item_product_seller_merged_path,
        )
    )
    # 5. 注文IDごとに期間の情報を結合
    df_order_payment_geolocation_seller_merged = _crete_span_cols(
        df=df_order_payment_geolocation_seller_merged
    )
    # 6. 注文IDごとに顧客と販売店の距離情報を結合
    df_order_payment_geolocation_seller_merged = (
        _create_distance_info_between_customer_and_seller(
            df=df_order_payment_geolocation_seller_merged
        )
    )
    if output_path is not None:
        df_order_payment_geolocation_seller_merged.to_csv(output_path, index=False)
    return df_order_payment_geolocation_seller_merged
