import json
import logging
import time
from typing import Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from transformers import pipeline

from .prompt import categorize_prompt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def _batch_translate_with_openai(texts: list[str], batch_size: int = 20) -> list[str]:
    """
    テキストのリストをバッチで翻訳する
    """
    client = OpenAI()
    translations = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        input_texts = [{"id": j, "text": text} for j, text in enumerate(batch)]
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a translator. Translate the following Portuguese texts to English. 
                        Respond with a JSON array where each object has 'id' and 'translation' fields. 
                        Maintain the original meaning and tone of the text.""",
                    },
                    {
                        "role": "user",
                        "content": f"Translate these texts: {json.dumps(input_texts, ensure_ascii=False)}",
                    },
                ],
                temperature=0.3,
            )
            try:
                content = response.choices[0].message.content
                # マークダウンの記号を削除
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]  # 最初の行を削除
                if content.endswith("```"):
                    content = content.rsplit("\n", 1)[0]  # 最後の行を削除
                if content.startswith("json"):
                    content = content.split("\n", 1)[1]  # jsonの行を削除
                content = content.strip()
                translated_batch = json.loads(content)
                translated_batch = sorted(translated_batch, key=lambda x: x["id"])
                translations.extend([item["translation"] for item in translated_batch])
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                print(f"Raw response: {response.choices[0].message.content}")
                print(f"Cleaned content: {content}")
                translations.extend(batch)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error in batch translation: {e}")
            translations.extend(batch)

    return translations


def translate_review_data(
    df: pd.DataFrame, column_name: str, max_rows: int = 1000
) -> pd.DataFrame:
    """
    データフレームの特定のカラムを翻訳する
    """
    df_copy = df.copy()
    non_null_mask = df[column_name].notna()
    texts_to_translate = df.loc[non_null_mask, column_name].head(max_rows).tolist()
    print(
        f"Starting translation for {len(texts_to_translate)} texts from {column_name}"
    )
    translations = _batch_translate_with_openai(texts_to_translate)
    new_column = f"{column_name}_en"
    df_copy[new_column] = df_copy[column_name]
    indices = df_copy.loc[non_null_mask].head(max_rows).index
    df_copy.loc[indices, new_column] = translations
    assert df_copy.shape[0] == df.shape[0]
    return df_copy


def categorize_positive_negative_review_data(
    df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """
    レビューデータを正負のカテゴリに分類する
    Nullの場合はNull値を返却する
    """
    sentiment_pipeline = pipeline(
        "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"
    )
    df_copy = df.copy()
    review_comment_message_en_list = df_copy[
        "review_comment_message_en"
    ].values.tolist()
    label_list = []
    score_list = []

    for review in tqdm(review_comment_message_en_list):
        # Nullチェック（None または np.nan）
        if pd.isna(review):
            label_list.append(None)
            score_list.append(None)
            continue
        categorize_result = sentiment_pipeline(review)
        score_list.append(categorize_result[0]["score"])
        if categorize_result[0]["score"] >= 0.70:
            label_list.append(categorize_result[0]["label"])
        else:
            label_list.append("not_clear")

    df_copy["label"] = label_list
    df_copy["label_translated"] = df_copy["label"].map(
        {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive",
            "not_clear": "not_clear",
            None: None,
        }
    )
    df_copy["label_score"] = score_list
    assert df_copy.shape[0] == df.shape[0]
    return df_copy


def modify_review_score_apply(row: pd.Series) -> float:
    label = row["label_translated"]
    score = row["review_score"]
    if label == "Negative":
        if score == 5.0:
            return 1.0
        elif score == 4.0:
            return 2.0
    elif label == "Positive":
        if score == 1.0:
            return 5.0
        elif score == 2.0:
            return 4.0
    return score


def _classify_review_sentiment(texts: str, batch_size: int = 20) -> list[str]:
    """
    レビューテキストの内容を分析し、主要な感想ポイントを分類する
    """
    client = OpenAI()
    classifications = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        input_texts = [{"id": j, "text": text} for j, text in enumerate(batch)]
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": categorize_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Please classify the following reviews and return the result in the specified JSON format: {json.dumps(input_texts, ensure_ascii=False)}",
                    },
                ],
                temperature=0.2,
            )
            try:
                content = response.choices[0].message.content
                # マークダウンの記号を削除
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                if content.startswith("json"):
                    content = content.split("\n", 1)[1]
                content = content.strip()
                classified_batch = json.loads(content)
                if not isinstance(classified_batch, list):
                    raise ValueError("Response is not a list")
                for item in classified_batch:
                    if "id" not in item or "categories" not in item:
                        raise ValueError("Invalid response format")
                classified_batch = sorted(classified_batch, key=lambda x: x["id"])
                # カテゴリーリストをJSON文字列に変換
                classifications.extend(
                    [json.dumps(item["categories"]) for item in classified_batch]
                )
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error processing response: {e}")
                print(f"Raw response: {response.choices[0].message.content}")
                classifications.extend([json.dumps(["Unclassifiable"]) for _ in batch])
            time.sleep(0.5)
        except Exception as e:
            print(f"API call error: {e}")
            classifications.extend([json.dumps(["Unclassifiable"]) for _ in batch])

    return classifications


def process_review_classification(
    df: pd.DataFrame,
    column_name: str = "review_comment_message_en",
    max_rows: int = 100,
):
    """
    データフレームの英訳済みレビューを分類する
    """
    df_copy = df.copy()
    non_null_mask = df_copy[column_name].notna()
    texts_to_classify = df_copy.loc[non_null_mask, column_name].head(max_rows).tolist()
    print(f"Starting classification for {len(texts_to_classify)} reviews")
    classifications = _classify_review_sentiment(texts_to_classify)
    df_copy["review_categories"] = None
    indices = df_copy.loc[non_null_mask].head(max_rows).index
    df_copy.loc[indices, "review_categories"] = classifications
    # カテゴリーを文字列からリストに戻す（必要な場合）
    df_copy["review_categories"] = df_copy["review_categories"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    assert df_copy.shape[0] == df.shape[0]
    return df_copy


def process_review_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    レビューデータの日付カラムを処理する
    """
    df_copy = df.copy()
    df_copy["review_creation_date"] = pd.to_datetime(df_copy["review_creation_date"])
    df_copy["review_creation_month"] = df_copy["review_creation_date"].dt.to_period("M")
    df_copy["review_answer_timestamp"] = pd.to_datetime(
        df_copy["review_answer_timestamp"]
    )
    df_copy["review_answer_month"] = df_copy["review_answer_timestamp"].dt.to_period(
        "M"
    )
    df_copy["review_answer_date"] = df_copy["review_answer_timestamp"].dt.to_period("D")
    return df_copy


def process_review_data(
    review_dataset_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    レビューデータを処理する
    """
    df_reviews = pd.read_csv(review_dataset_path)
    df_reviews_copy = df_reviews.copy()
    df_reviews_unique = df_reviews[
        [
            "review_id",
            "review_comment_title",
            "review_comment_message",
            "review_creation_date",
            "review_score",
        ]
    ].drop_duplicates()
    # レビュー内容を英訳する
    logger.info("Translation task start")
    df_reviews_translated = translate_review_data(
        df=df_reviews_unique,
        column_name="review_comment_title",
        max_rows=df_reviews_unique.shape[0],
    )
    df_reviews_translated = translate_review_data(
        df=df_reviews_translated,
        column_name="review_comment_message",
        max_rows=df_reviews_translated.shape[0],
    )
    logger.info("Translation task end")
    logger.info("Sentiment analysis task start")
    # レビュー内容を正負のカテゴリに分類する
    df_reviews_translated_labelled = categorize_positive_negative_review_data(
        df=df_reviews_translated,
        column_name="review_comment_message_en",
    )
    # レビュースコアを逆に記入したものを修正
    df_reviews_translated_labelled["modified_review_score"] = (
        df_reviews_translated_labelled.apply(modify_review_score_apply, axis=1)
    )
    logger.info("Sentiment analysis task end")
    logger.info("Classification task start")
    # レビュー内容を分類する
    df_reviews_translated_labelled_classified = process_review_classification(
        df=df_reviews_translated_labelled,
        max_rows=df_reviews_translated_labelled.shape[0],
    )
    df_reviews_translated_labelled_classified["review_categories_str"] = (
        df_reviews_translated_labelled_classified["review_categories"].apply(
            lambda x: "_".join(x) if isinstance(x, list) else None
        )
    )
    df_reviews_translated_labelled_classified["review_categories_str_modified"] = (
        df_reviews_translated_labelled_classified["review_categories_str"].apply(
            lambda x: x.replace(" Praise", "_Praise").replace(" Issue", "_Issue")
            if type(x) is str
            else None
        )
    )
    logger.info("Classification task end")
    df_reviews_processed = pd.merge(
        df_reviews_copy,
        df_reviews_translated_labelled_classified[
            [
                "review_id",
                "review_comment_title_en",
                "review_comment_message_en",
                "label_translated",
                "label_score",
                "modified_review_score",
                "review_categories",
                "review_categories_str",
                "review_categories_str_modified",
            ]
        ],
        on="review_id",
        how="left",
    )
    assert df_reviews_processed.shape[0] == df_reviews_copy.shape[0]
    df_reviews_processed_added_date_cols = process_review_date_cols(
        df=df_reviews_processed
    )
    if output_path:
        df_reviews_processed_added_date_cols.to_csv(output_path, index=False)
    return df_reviews_processed_added_date_cols
