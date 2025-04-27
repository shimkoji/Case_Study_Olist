import pandas as pd


def agg_review_count_by_category_name(
    df: pd.DataFrame, product_category_name: list[str]
) -> dict[str, pd.DataFrame]:
    """
    Aggregates and summarizes review counts by category for the specified product categories.

    Args:
        df (pd.DataFrame): The input DataFrame containing review data.
        product_category_name (list[str]): List of product category names to filter the DataFrame.

    Returns:
        dict[str, pd.DataFrame]: A dictionary with keys 'all', 'praise', and 'issue', each containing
            a DataFrame with the count and percentage of reviews for each review category.
            - 'all': Statistics for all review categories.
            - 'praise': Statistics for categories ending with 'Praise'.
            - 'issue': Statistics for categories ending with 'Issue'.

    The function prints summary statistics and returns the aggregated results.
    """
    result = {}
    df_target = df.query("product_category_name_english in @product_category_name")
    df_target_praise = df_target.query(
        "review_categories_str_modified.str.endswith('Praise')"
    )
    df_target_issue = df_target.query(
        "review_categories_str_modified.str.endswith('Issue')"
    )
    print(f"Praise Review num:{df_target_praise['review_id'].nunique()}")
    print(f"Praise Issue num:{df_target_issue['review_id'].nunique()}")
    print(
        f"Issue ratio:{(df_target_issue['review_id'].nunique() / df_target['review_id'].nunique()):.2f}"
    )
    category_count_stats = (
        df_target.groupby(["review_categories_str_modified"])["review_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
    )
    category_count_stats["percentage"] = (
        category_count_stats["review_id"] / category_count_stats["review_id"].sum()
    ).round(2)
    category_count_stats.columns = ["category", "count", "percentage"]
    print("Overall")
    print(category_count_stats.head(20))
    print("praise")
    category_praise_count_stats = (
        df_target.query("review_categories_str_modified.str.endswith('Praise')")
        .groupby(["review_categories_str_modified"])["review_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
    )
    category_praise_count_stats["percentage"] = (
        category_praise_count_stats["review_id"]
        / category_praise_count_stats["review_id"].sum()
    ).round(2)
    category_praise_count_stats.columns = ["category", "count", "percentage"]
    print(category_praise_count_stats.head(20))
    print("issue")
    category_issue_count_stats = (
        df_target.query("review_categories_str_modified.str.endswith('Issue')")
        .groupby(["review_categories_str_modified"])["review_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
    )
    category_issue_count_stats["percentage"] = (
        category_issue_count_stats["review_id"]
        / category_issue_count_stats["review_id"].sum()
    ).round(2)
    category_issue_count_stats.columns = ["category", "count", "percentage"]
    print(category_issue_count_stats.head(20))
    result["all"] = category_count_stats
    result["praise"] = category_praise_count_stats
    result["issue"] = category_issue_count_stats
    return result


def agg_review_category_ratio_by_category_name(
    df: pd.DataFrame, product_category_name: list[str]
) -> None:
    """
    Prints the ratio and summary statistics of review categories for the specified product categories.

    Args:
        df (pd.DataFrame): The input DataFrame containing review data.
        product_category_name (list[str]): List of product category names to filter the DataFrame.

    Returns:
        None

    The function prints the number of unique reviews for 'Praise' and 'Issue' categories,
    the issue ratio, and the top categories with their counts and percentages.
    """
    df_target = df.query("product_category_name_english in @product_category_name")
    df_target_praise = df_target.query(
        "review_categories_str_modified.str.endswith('Praise')"
    )
    df_target_issue = df_target.query(
        "review_categories_str_modified.str.endswith('Issue')"
    )
    print(f"Praise Review num:{df_target_praise['review_id'].nunique()}")
    print(f"Praise Issue num:{df_target_issue['review_id'].nunique()}")
    print(
        f"Issue ratio:{(df_target_issue['review_id'].nunique() / df_target['review_id'].nunique()):.2f}"
    )
    category_count_stats = (
        df_target.groupby(["review_categories_str_modified"])["review_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
    )
    category_count_stats["percentage"] = (
        category_count_stats["review_id"] / category_count_stats["review_id"].sum()
    ).round(2)
    category_count_stats.columns = ["category", "count", "percentage"]
    print("Overall")
    print(category_count_stats.head(20))
    print("praise")
    category_praise_count_stats = (
        df_target.query("review_categories_str_modified.str.endswith('Praise')")
        .groupby(["review_categories_str_modified"])["review_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
    )
    category_praise_count_stats["percentage"] = (
        category_praise_count_stats["review_id"]
        / category_praise_count_stats["review_id"].sum()
    ).round(2)
    category_praise_count_stats.columns = ["category", "count", "percentage"]
    print(category_praise_count_stats.head(20))
    print("issue")
    category_issue_count_stats = (
        df_target.query("review_categories_str_modified.str.endswith('Issue')")
        .groupby(["review_categories_str_modified"])["review_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
    )
    category_issue_count_stats["percentage"] = (
        category_issue_count_stats["review_id"]
        / category_issue_count_stats["review_id"].sum()
    ).round(2)
    category_issue_count_stats.columns = ["category", "count", "percentage"]
    print(category_issue_count_stats.head(20))
