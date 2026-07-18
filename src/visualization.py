import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from typing import Dict, Optional


CLUSTER_COLORS = ["#e82135", "#1976d2", "#43a047", "#8e44cc", "#fb8c00", "#00acc1", "#fdd835", "#ec407a"]
PAPER_BG = "rgba(0,0,0,0)"
PLOT_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(222, 235, 255, 0.10)"
FONT_COLOR = "#f3f6fb"
MUTED_COLOR = "#aab7c8"


def _cluster_name(cluster_id: int) -> str:
    names = {
        0: "Action Lovers",
        1: "Drama Seekers",
        2: "Comedy Fans",
        3: "Documentary Lovers",
        4: "Family Viewers",
        5: "Global Explorers",
        6: "Classic Watchers",
        7: "TV Series Fans",
    }
    return names.get(int(cluster_id), f"Cụm {int(cluster_id) + 1}")


def _base_layout(fig: go.Figure, height: int = 360, showlegend: bool = False, margin: Optional[Dict] = None) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, family="Inter, Arial, sans-serif", size=12),
        margin=margin or dict(l=20, r=20, t=28, b=28),
        showlegend=showlegend,
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=FONT_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=FONT_COLOR),
        legend=dict(
            bgcolor="rgba(8, 22, 37, 0.65)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(color=FONT_COLOR),
        ),
    )
    return fig


def fig_cluster_distribution(labels: pd.Series) -> go.Figure:
    counts = pd.Series(labels).value_counts().sort_index()
    total = int(counts.sum())
    display_labels = [f"Cụm {int(cluster) + 1} - {_cluster_name(cluster)}" for cluster in counts.index]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=display_labels,
                values=counts.values,
                hole=0.48,
                sort=False,
                domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
                marker=dict(colors=CLUSTER_COLORS[: len(counts)], line=dict(color="#07111f", width=2)),
                textinfo="percent",
                textposition="inside",
                textfont=dict(color="#ffffff", size=13),
                hovertemplate="%{label}<br>%{value:,} nội dung (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        annotations=[
            dict(
                text=f"<b>{total:,}</b><br>nội dung",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color=FONT_COLOR),
            )
        ],
    )
    return _base_layout(fig, height=360, showlegend=False)


def fig_cluster_profile_terms(df: pd.DataFrame, labels: pd.Series, top_n: int = 9) -> go.Figure:
    df = df.assign(cluster=labels)
    exploded = df[["cluster", "genre_list"]].explode("genre_list")
    counts = exploded.groupby(["genre_list", "cluster"]).size().reset_index(name="count")
    top_genres = exploded["genre_list"].value_counts().head(top_n).index.tolist()
    cluster_ids = sorted(pd.Series(labels).unique())
    matrix = []
    for genre in top_genres:
        row = []
        for cluster_id in cluster_ids:
            value = counts.loc[
                (counts["genre_list"] == genre) & (counts["cluster"] == cluster_id),
                "count",
            ].sum()
            row.append(int(value))
        row_total = max(sum(row), 1)
        matrix.append([round(value / row_total, 3) for value in row])

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"C{int(cluster_id) + 1}" for cluster_id in cluster_ids],
            y=top_genres,
            colorscale=[
                [0.0, "#241b3d"],
                [0.25, "#7e174c"],
                [0.55, "#e53034"],
                [0.78, "#ff8f1f"],
                [1.0, "#ffd166"],
            ],
            colorbar=dict(title="Tỷ lệ", ticktext=["Thấp", "Cao"], tickvals=[0, 1]),
            hovertemplate="%{y}<br>%{x}: %{z:.1%}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_xaxes(side="top", showticklabels=False, title="Cụm", automargin=True)
    fig.update_yaxes(tickfont=dict(size=11), automargin=True)
    return _base_layout(fig, height=360, margin=dict(l=145, r=55, t=48, b=20))


def fig_cluster_profile_countries(df: pd.DataFrame, labels: pd.Series, top_n: int = 6) -> go.Figure:
    df = df.assign(cluster=labels)
    exploded = df[["cluster", "country_list"]].explode("country_list")
    counts = exploded.groupby(["cluster", "country_list"]).size().reset_index(name="count")
    top = counts.groupby("cluster").apply(lambda group: group.nlargest(top_n, "count")).reset_index(drop=True)
    if top.empty:
        return go.Figure()
    fig = go.Figure()
    for cluster_id in sorted(top["cluster"].unique()):
        subset = top[top["cluster"] == cluster_id].sort_values("count")
        fig.add_trace(
            go.Bar(
                x=subset["count"],
                y=subset["country_list"],
                orientation="h",
                name=f"Cụm {int(cluster_id) + 1}",
                marker_color=CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)],
                hovertemplate="%{y}: %{x:,}<extra></extra>",
            )
        )
    return _base_layout(fig, height=300, showlegend=True)


def fig_pca_scatter(feature_matrix: pd.DataFrame, labels: pd.Series) -> go.Figure:
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(feature_matrix)
    scatter = pd.DataFrame({"pc1": components[:, 0], "pc2": components[:, 1], "cluster": labels})
    explained = pca.explained_variance_ratio_ * 100
    total_explained = explained.sum()
    info_loss = 100 - total_explained

    fig = go.Figure()
    for cluster_id in sorted(scatter["cluster"].unique()):
        subset = scatter[scatter["cluster"] == cluster_id]
        fig.add_trace(
            go.Scattergl(
                x=subset["pc1"],
                y=subset["pc2"],
                mode="markers",
                name=f"Cụm {int(cluster_id) + 1}",
                marker=dict(
                    color=CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)],
                    size=5,
                    opacity=0.84,
                    line=dict(width=0),
                ),
                hovertemplate="PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
            )
        )
    fig.update_xaxes(title=f"PC1 ({explained[0]:.1f}%)", automargin=True)
    fig.update_yaxes(title=f"PC2 ({explained[1]:.1f}%)", automargin=True)
    fig.update_layout(
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
        annotations=[
            dict(
                text=f"Giữ lại <b>{total_explained:.1f}%</b> | Mất mát <b style='color:#e82135'>{info_loss:.1f}%</b>",
                xref="paper", yref="paper",
                x=0.01, y=0.01,
                showarrow=False,
                font=dict(size=11, color=MUTED_COLOR),
                align="left",
            )
        ],
    )
    return _base_layout(fig, height=360, showlegend=True)


def fig_top_genres_bar(df: pd.DataFrame, top_n: int = 9) -> go.Figure:
    counts = df["genre_list"].explode().value_counts().head(top_n).sort_values()
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.values,
                y=counts.index,
                orientation="h",
                marker=dict(color="#1e73d8", line=dict(color="#3694ff", width=1)),
                text=[f"{value:,}" for value in counts.values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: %{x:,} lượt xem quy đổi<extra></extra>",
            )
        ]
    )
    fig.update_xaxes(title="Lượt xem", tickformat="~s", automargin=True)
    fig.update_yaxes(automargin=True)
    return _base_layout(fig, height=330, margin=dict(l=145, r=40, t=25, b=45))


def fig_silhouette_by_cluster(feature_matrix: pd.DataFrame, labels: pd.Series) -> go.Figure:
    label_series = pd.Series(labels)
    if label_series.nunique() <= 1 or len(label_series) < 3:
        values = label_series.value_counts().sort_index() * 0
    else:
        sample_size = min(2000, len(feature_matrix))
        sampled_matrix = feature_matrix.sample(n=sample_size, random_state=42)
        sampled_labels = label_series.loc[sampled_matrix.index]
        samples = silhouette_samples(sampled_matrix, sampled_labels)
        values = pd.DataFrame({"cluster": sampled_labels, "score": samples}).groupby("cluster")["score"].mean()
        values = values.reindex(label_series.value_counts().sort_index().index).fillna(0)

    fig = go.Figure(
        data=[
            go.Bar(
                x=values.values,
                y=[f"Cụm {int(cluster_id) + 1}" for cluster_id in values.index],
                orientation="h",
                marker_color=[CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)] for cluster_id in values.index],
                text=[f"{value:.2f}" for value in values.values],
                textposition="outside",
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_xaxes(range=[0, max(1, float(values.max()) + 0.1)], title="")
    return _base_layout(fig, height=230)
