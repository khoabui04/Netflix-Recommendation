import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler

from src.data_preprocessing import preprocess_netflix_data
from src.feature_engineering import build_feature_matrix


def build_clusters(feature_matrix: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(feature_matrix)
    return labels, model


def evaluate_clusters(feature_matrix: pd.DataFrame, labels: pd.Series) -> tuple[float, float]:
    """Trả về (silhouette_score, davies_bouldin_score).

    Silhouette: cao hơn tốt hơn (max 1).
    Davies-Bouldin: thấp hơn tốt hơn (min 0).
    """
    n_unique = len(set(labels))
    if n_unique <= 1:
        return 0.0, 0.0
    sample_size = min(2000, len(feature_matrix))
    sil = silhouette_score(feature_matrix, labels, sample_size=sample_size, random_state=42)
    db = davies_bouldin_score(feature_matrix, labels)
    return float(sil), float(db)


def summarize_clusters(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    summary = []
    grouped = df.assign(cluster=labels).groupby("cluster")
    for cluster_id, group in grouped:
        top_genres = group["genre_list"].explode().value_counts().head(3).index.tolist()
        top_countries = group["country_list"].explode().value_counts().head(3).index.tolist()
        top_type = group["type"].value_counts().idxmax()
        summary.append(
            {
                "cluster": cluster_id,
                "size": len(group),
                "avg_release_year": int(group["release_year"].mean()),
                "avg_duration_min": int(group["duration_min"].mean()),
                "top_type": top_type,
                "top_genres": ", ".join(top_genres),
                "top_countries": ", ".join(top_countries),
            }
        )
    return pd.DataFrame(summary).set_index("cluster").sort_index()


def build_cluster_profiles(df: pd.DataFrame, labels: pd.Series, n_top: int = 3) -> pd.DataFrame:
    rows = []
    grouped = df.assign(cluster=labels).groupby("cluster")
    for cluster_id, group in grouped:
        top_genres = group["genre_list"].explode().value_counts().head(n_top).index.tolist()
        top_countries = group["country_list"].explode().value_counts().head(n_top).index.tolist()
        top_type = group["type"].value_counts().idxmax()
        rows.append(
            {
                "cluster": cluster_id,
                "size": len(group),
                "main_type": top_type,
                "common_genres": ", ".join(top_genres),
                "common_countries": ", ".join(top_countries),
                "avg_duration_min": int(group["duration_min"].mean()),
            }
        )
    return pd.DataFrame(rows).set_index("cluster").sort_index()


def predict_new_title(
    title: str,
    content_type: str,
    listed_in: str,
    country: str,
    duration: str,
    rating: str,
    release_year: int,
    kmeans_model: KMeans,
    scaler: MinMaxScaler,
    top_genre_list: list,
    top_country_list: list,
) -> int:
    """Phân loại một title mới vào cluster đã học.

    Dùng lại scaler, top_genre_list, top_country_list từ lúc training
    để đảm bảo feature vector nhất quán.

    Returns: cluster_id (int)
    """
    raw = pd.DataFrame(
        [
            {
                "show_id": "new_0",
                "type": content_type,
                "title": title,
                "director": "Unknown",
                "country": country,
                "release_year": release_year,
                "rating": rating,
                "duration": duration,
                "listed_in": listed_in,
            }
        ]
    )
    processed = preprocess_netflix_data(raw)
    feature_matrix, _, _, _, _ = build_feature_matrix(
        processed,
        top_genre_list=top_genre_list,
        top_country_list=top_country_list,
        scaler=scaler,
    )
    cluster_id = int(kmeans_model.predict(feature_matrix)[0])
    return cluster_id
