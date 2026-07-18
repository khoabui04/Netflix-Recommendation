import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Tuple


def get_top_terms(series: pd.Series, top_n: int) -> List:
    counts = {}
    for values in series:
        for value in values:
            counts[value] = counts.get(value, 0) + 1
    sorted_terms = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [term for term, _ in sorted_terms[:top_n]]


def build_feature_matrix(
    df: pd.DataFrame,
    top_genres: int = 15,
    top_countries: int = 10,
    top_genre_list: Optional[List] = None,
    top_country_list: Optional[List] = None,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[pd.DataFrame, List, List, List, MinMaxScaler]:
    """Build feature matrix.

    Khi top_genre_list / top_country_list / scaler được truyền vào (predict mode),
    hàm dùng lại chúng thay vì fit mới — đảm bảo consistency với training data.

    Returns: (feature_matrix, feature_columns, top_genre_list, top_country_list, scaler)
    """
    df = df.copy()

    # is_movie duy nhất — is_tvshow tương quan -1 nên bỏ để tránh redundancy
    df["is_movie"] = (df["type"] == "Movie").astype(int)

    if top_genre_list is None:
        top_genre_list = get_top_terms(df["genre_list"], top_genres)
    for genre in top_genre_list:
        df[f"genre_{genre}"] = df["genre_list"].apply(lambda values: int(genre in values))

    if top_country_list is None:
        top_country_list = get_top_terms(df["country_list"], top_countries)
    for country in top_country_list:
        df[f"country_{country}"] = df["country_list"].apply(lambda values: int(country in values))

    df["duration_min_scaled"] = df["duration_min"].astype(float)
    df["season_count_scaled"] = df["season_count"].astype(float)
    df["rating_level_scaled"] = df["rating_level"].astype(float)

    numeric_cols = ["duration_min_scaled", "season_count_scaled", "rating_level_scaled"]
    if scaler is None:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    feature_columns = ["is_movie"] + numeric_cols
    feature_columns += [f"genre_{genre}" for genre in top_genre_list]
    feature_columns += [f"country_{country}" for country in top_country_list]

    feature_matrix = df[feature_columns].copy()
    return feature_matrix, feature_columns, top_genre_list, top_country_list, scaler
