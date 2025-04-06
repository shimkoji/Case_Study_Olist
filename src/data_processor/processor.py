from typing import Optional

import pandas as pd

from .merge_order_customer import process_order_customer_data
from .merge_order_details import process_order_details_data
from .merge_products_seller import process_products_seller_data
from .process_review_data import process_review_data


class Processor:
    def __init__(
        self,
        raw_data_path: str,
        processed_data_path: str,
        interim_data_path: str,
    ) -> None:
        self.raw_data_path = raw_data_path
        self.interim_data_path = interim_data_path
        self.processed_data_path = processed_data_path
        pass

    def merge_products_seller_data(
        self, output_path: Optional[str] = None
    ) -> pd.DataFrame:
        df_order_items_products_sellers_merged = process_products_seller_data(
            product_dataset_path=self.raw_data_path / "olist_products_dataset.csv",
            category_name_dataset_path=self.raw_data_path
            / "product_category_name_translation.csv",
            order_item_dataset_path=self.raw_data_path
            / "olist_order_items_dataset.csv",
            seller_dataset_path=self.raw_data_path / "olist_sellers_dataset.csv",
            output_path=output_path,
        )
        return df_order_items_products_sellers_merged

    def merge_order_customer_data(
        self, output_path: Optional[str] = None
    ) -> pd.DataFrame:
        df_orders_customer_merged = process_order_customer_data(
            customer_dataset_path=self.raw_data_path / "olist_customers_dataset.csv",
            orders_dataset_path=self.raw_data_path / "olist_orders_dataset.csv",
            output_path=output_path,
        )
        return df_orders_customer_merged

    def process_review_data(self, output_path: Optional[str] = None) -> pd.DataFrame:
        df_reviews_copy_translated_labelled_classified_merged = process_review_data(
            review_dataset_path=self.raw_data_path / "olist_order_reviews_dataset.csv",
            output_path=output_path,
        )

        return df_reviews_copy_translated_labelled_classified_merged

    def process_order_details_data(
        self, output_path: Optional[str] = None
    ) -> pd.DataFrame:
        df_order_payment_geolocation_seller_merged = process_order_details_data(
            order_items_dataset_path=self.interim_data_path
            / "olist_orders_customer_merged.csv",
            geolocation_dataset_path=self.raw_data_path
            / "olist_geolocation_dataset.csv",
            payments_dataset_path=self.raw_data_path
            / "olist_order_payments_dataset.csv",
            item_product_seller_merged_path=self.interim_data_path
            / "olist_item_product_seller_merged.csv",
            output_path=output_path,
        )
        return df_order_payment_geolocation_seller_merged
