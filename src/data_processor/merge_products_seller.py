from typing import Optional

import pandas as pd


def process_products_seller_data(
    product_dataset_path: str,
    category_name_dataset_path: str,
    order_item_dataset_path: str,
    seller_dataset_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Process the product, category name, order item, and seller data to create a dataset with product information, seller information, and order item information.
    """
    df_products = pd.read_csv(product_dataset_path)
    df_category_name = pd.read_csv(category_name_dataset_path)
    df_order_items = pd.read_csv(order_item_dataset_path)
    df_sellers = pd.read_csv(seller_dataset_path)
    df_prodcuts_merged = pd.merge(
        df_products,
        df_category_name,
        on="product_category_name",
        how="left",
    )
    assert df_prodcuts_merged.shape[0] == df_products.shape[0]
    df_order_items["sum_price_freight_by_order"] = (
        df_order_items["price"] + df_order_items["freight_value"]
    )
    df_order_items_products_merged = pd.merge(
        df_order_items, df_prodcuts_merged, on="product_id", how="left"
    )
    assert df_order_items.shape[0] == df_order_items_products_merged.shape[0]
    df_order_items_products_sellers_merged = pd.merge(
        df_order_items_products_merged, df_sellers, on="seller_id", how="left"
    )
    assert (
        df_order_items_products_sellers_merged.shape[0]
        == df_order_items_products_merged.shape[0]
    )
    if output_path is not None:
        df_order_items_products_sellers_merged.to_csv(output_path, index=False)
    return df_order_items_products_sellers_merged
