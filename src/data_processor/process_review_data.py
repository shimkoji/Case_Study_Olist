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
    Translates a list of texts in batches using OpenAI's API.

    Args:
        texts (list[str]): A list of strings to be translated.
        batch_size (int): The number of texts to process in each batch.

    Returns:
        list[str]: A list of translated strings.
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
                # Remove markdown symbols
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]  # Remove the first line
                if content.endswith("```"):
                    content = content.rsplit("\n", 1)[0]  # Remove the last line
                if content.startswith("json"):
                    content = content.split("\n", 1)[1]  # Remove the json line
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
    Translates a specific column in a DataFrame using OpenAI's API.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to translate.
        max_rows (int): The maximum number of rows to translate.

    Returns:
        pd.DataFrame: A new DataFrame with an additional column containing the translated text.
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
    Categorizes review data into positive, negative, or neutral sentiment categories using a pre-trained sentiment analysis model.
    Returns Null values for null inputs.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column containing the review text to categorize.

    Returns:
        pd.DataFrame: A new DataFrame with added 'label' and 'label_translated' columns
                    representing the sentiment category and a 'label_score' column representing the confidence score.
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
    """
    Modifies the review score based on the sentiment label.

    Args:
        row (pd.Series): A row from the DataFrame containing 'label_translated' and 'review_score' columns.

    Returns:
        float: The modified review score.
    """
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
    Analyzes the content of review texts and classifies the main feedback points.

    Args:
        texts (str): A list of review texts to be classified.
        batch_size (int): The number of texts to process in each batch.

    Returns:
        list[str]: A list of JSON strings, where each string represents a list of categories
                    assigned to each review text.
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
                # Remove markdown symbols
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
                # Convert the category list to a JSON string
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
    Classifies the English-translated reviews in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column containing the English-translated review texts.
                        Defaults to "review_comment_message_en".
        max_rows (int): The maximum number of rows to classify. Defaults to 100.

    Returns:
        pd.DataFrame: A new DataFrame with an added 'review_categories' column containing
                    the classification results as a list of categories.
    """
    df_copy = df.copy()
    non_null_mask = df_copy[column_name].notna()
    texts_to_classify = df_copy.loc[non_null_mask, column_name].head(max_rows).tolist()
    print(f"Starting classification for {len(texts_to_classify)} reviews")
    classifications = _classify_review_sentiment(texts_to_classify)
    df_copy["review_categories"] = None
    indices = df_copy.loc[non_null_mask].head(max_rows).index
    df_copy.loc[indices, "review_categories"] = classifications
    # Convert the category back from string to list
    df_copy["review_categories"] = df_copy["review_categories"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    assert df_copy.shape[0] == df.shape[0]
    return df_copy


def process_review_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the date columns in the review data.

    Args:
        df (pd.DataFrame): The input DataFrame containing review data with date columns.

    Returns:
        pd.DataFrame: A new DataFrame with added 'review_creation_month', 'review_answer_month',
                    and 'review_answer_date' columns derived from the original date columns.
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
    Processes review data by translating, categorizing sentiment, classifying content,
    and adding date-related columns.

    Args:
        review_dataset_path (str): Path to the review dataset CSV file.
        output_path (Optional[str]): Optional path to save the processed DataFrame as a CSV file.

    Returns:
        pd.DataFrame: A processed DataFrame containing translated review texts, sentiment labels,
                    content classifications, and date-related features.
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
    # Translate review content to English
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
    # Categorize review content into positive/negative sentiment
    df_reviews_translated_labelled = categorize_positive_negative_review_data(
        df=df_reviews_translated,
        column_name="review_comment_message_en",
    )
    # Fix reversed review scores
    df_reviews_translated_labelled["modified_review_score"] = (
        df_reviews_translated_labelled.apply(modify_review_score_apply, axis=1)
    )
    logger.info("Sentiment analysis task end")
    logger.info("Classification task start")
    # Classify review content
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
