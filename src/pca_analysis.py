import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple


PAPER_BG = "rgba(0,0,0,0)"
PLOT_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(222, 235, 255, 0.10)"
FONT_COLOR = "#f3f6fb"


def compute_pca_variance(feature_matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Fit PCA với tất cả components, trả về explained + cumulative variance ratio."""
    pca = PCA(n_components=None, random_state=42)
    pca.fit(feature_matrix)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    return explained, cumulative


def recommend_n_components(cumulative: np.ndarray, thresholds: Optional[List[float]] = None) -> Dict[float, int]:
    """Trả về số components tối thiểu để đạt mỗi ngưỡng variance."""
    if thresholds is None:
        thresholds = [0.70, 0.80, 0.90, 0.95]
    result = {}
    for t in thresholds:
        n = int(np.argmax(cumulative >= t)) + 1
        result[t] = n
    return result


def fig_scree_plot(feature_matrix: pd.DataFrame) -> Tuple[go.Figure, go.Figure, Dict[float, int]]:
    """Tạo scree plot và cumulative variance chart.

    Returns: (fig_scree, fig_cumulative, recommendations)
    """
    explained, cumulative = compute_pca_variance(feature_matrix)
    recommendations = recommend_n_components(cumulative)
    n_total = len(explained)
    x_vals = list(range(1, n_total + 1))

    # --- Scree plot ---
    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(
        x=x_vals,
        y=(explained * 100).tolist(),
        marker_color="#1e73d8",
        hovertemplate="PC%{x}: %{y:.1f}%<extra></extra>",
        name="Variance mỗi PC",
    ))
    fig_scree.update_layout(
        height=320,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, family="Inter, Arial, sans-serif", size=12),
        margin=dict(l=20, r=20, t=32, b=28),
        showlegend=False,
        xaxis=dict(title="Principal Component", gridcolor=GRID_COLOR, color=FONT_COLOR),
        yaxis=dict(title="Explained Variance (%)", gridcolor=GRID_COLOR, color=FONT_COLOR),
    )

    # --- Cumulative variance ---
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=x_vals,
        y=(cumulative * 100).tolist(),
        mode="lines+markers",
        line=dict(color="#e82135", width=2),
        marker=dict(size=4),
        hovertemplate="PC1–%{x}: %{y:.1f}%<extra></extra>",
        name="Cumulative variance",
    ))
    # Đường ngưỡng tham chiếu
    threshold_styles = {0.80: ("#fb8c00", "80%"), 0.90: ("#43a047", "90%"), 0.95: ("#9553e6", "95%")}
    for thr, (color, label) in threshold_styles.items():
        n_rec = recommendations.get(thr, None)
        fig_cum.add_hline(
            y=thr * 100,
            line=dict(color=color, dash="dash", width=1),
            annotation_text=f"{label} → {n_rec} PCs",
            annotation_position="right",
            annotation_font=dict(color=color, size=11),
        )
    fig_cum.update_layout(
        height=320,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, family="Inter, Arial, sans-serif", size=12),
        margin=dict(l=20, r=80, t=32, b=28),
        showlegend=False,
        xaxis=dict(title="Số components", gridcolor=GRID_COLOR, color=FONT_COLOR),
        yaxis=dict(title="Cumulative Variance (%)", gridcolor=GRID_COLOR, color=FONT_COLOR),
    )

    return fig_scree, fig_cum, recommendations
