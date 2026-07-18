import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def build_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    missing_data = [
        {"feature": col, "missing_count": int(df[col].isna().sum())}
        for col in ["country", "director", "listed_in", "duration", "rating"]
        if col in df.columns
    ]
    return pd.DataFrame(missing_data)


def fig_type_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["type"].value_counts().reset_index()
    counts.columns = ["type", "count"]
    fig = px.bar(
        counts,
        x="type",
        y="count",
        color="type",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Phân phối loại nội dung",
        labels={"type": "Loại", "count": "Số lượng"},
        hover_data={"type": False, "count": True},
        template="plotly_dark",
    )
    fig.update_layout(showlegend=False)
    return fig


def fig_top_terms(series: pd.Series, top_n: int = 10, title: str = "Top Thể loại") -> go.Figure:
    counts = {}
    for values in series:
        for item in values:
            counts[item] = counts.get(item, 0) + 1
    sorted_terms = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not sorted_terms:
        return go.Figure()
    labels, values = zip(*sorted_terms)
    fig = px.bar(
        x=list(values),
        y=list(labels),
        orientation="h",
        title=title,
        labels={"x": "Số lượng", "y": "Thể loại"},
        color=list(values),
        color_continuous_scale="Viridis",
        template="plotly_dark",
    )
    fig.update_traces(hovertemplate="%{y}: %{x}<extra></extra>")
    fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"))
    return fig


def fig_duration_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="duration_min",
        nbins=25,
        title="Phân phối thời lượng (phút, cắt tối đa 250)",
        labels={"duration_min": "Thời lượng (phút)", "count": "Số lượng"},
        range_x=[0, 250],
        color_discrete_sequence=["#4c72b0"],
        template="plotly_dark",
    )
    fig.update_traces(hovertemplate="%{x} phút: %{y} tiêu đề<extra></extra>")
    return fig


def fig_rating_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["rating"].value_counts().reset_index()
    counts.columns = ["rating", "count"]
    fig = px.bar(
        counts,
        x="count",
        y="rating",
        orientation="h",
        title="Phân phối xếp hạng",
        labels={"rating": "Xếp hạng", "count": "Số lượng"},
        color="count",
        color_continuous_scale="RdBu",
        template="plotly_dark",
    )
    fig.update_traces(hovertemplate="%{y}: %{x} tiêu đề<extra></extra>")
    fig.update_layout(showlegend=False)
    return fig


def fig_country_distribution(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    exploded = df["country_list"].explode()
    top = exploded.value_counts().head(top_n).reset_index()
    top.columns = ["country", "count"]
    fig = px.bar(
        top,
        x="count",
        y="country",
        orientation="h",
        title=f"Top {top_n} quốc gia trong dữ liệu",
        labels={"country": "Quốc gia", "count": "Số lượng"},
        color="count",
        color_continuous_scale="ice",
        template="plotly_dark",
    )
    fig.update_traces(hovertemplate="%{y}: %{x} tiêu đề<extra></extra>")
    fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"))
    return fig


def fig_release_year(df: pd.DataFrame) -> go.Figure:
    """Phân phối nội dung theo năm phát hành, tách Movie / TV Show."""
    year_counts = (
        df[df["release_year"] > 0]
        .groupby(["release_year", "type"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        year_counts,
        x="release_year",
        y="count",
        color="type",
        barmode="stack",
        title="Phân phối nội dung theo năm phát hành",
        labels={"release_year": "Năm", "count": "Số lượng", "type": "Loại"},
        color_discrete_sequence=["#e82135", "#1976d2"],
        template="plotly_dark",
    )
    fig.update_traces(hovertemplate="%{x}: %{y}<extra></extra>")
    fig.update_layout(showlegend=True, legend_title_text="")
    return fig


def fig_correlation_heatmap(feature_matrix: pd.DataFrame) -> go.Figure:
    """Heatmap tương quan giữa các features số trong feature matrix.

    Chỉ hiển thị các cột numeric (loại bỏ binary genre/country có variance thấp).
    """
    numeric_cols = [c for c in feature_matrix.columns if not c.startswith("genre_") and not c.startswith("country_")]
    if len(numeric_cols) < 2:
        numeric_cols = list(feature_matrix.columns)

    corr = feature_matrix[numeric_cols].corr().round(2)
    labels = corr.columns.tolist()

    # Mask tam giác trên để tránh thông tin trùng lặp
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    z = corr.values.copy().astype(float)
    z[mask] = np.nan

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale=[
                [0.0, "#1565c0"],
                [0.5, "#f5f5f5"],
                [1.0, "#c62828"],
            ],
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in z],
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(title="r", tickvals=[-1, -0.5, 0, 0.5, 1]),
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3f6fb", family="Inter, Arial, sans-serif", size=11),
        margin=dict(l=100, r=20, t=32, b=80),
        xaxis=dict(tickangle=-35, color="#f3f6fb"),
        yaxis=dict(autorange="reversed", color="#f3f6fb"),
        title="Ma trận tương quan (features số)",
    )
    return fig
