import pandas as pd


def agg_review_category_ratio_by_category_name(
    df: pd.DataFrame, product_category_name: list[str]
) -> dict[str, pd.DataFrame]:
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
    print("全体")
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
