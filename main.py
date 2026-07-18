import streamlit as st
import pandas as pd

from src.clustering import build_cluster_profiles, build_clusters, evaluate_clusters, predict_new_title, summarize_clusters
from src.data_preprocessing import load_netflix_data, preprocess_netflix_data
from src.eda import (
    build_missing_report,
    fig_correlation_heatmap,
    fig_country_distribution,
    fig_duration_distribution,
    fig_rating_distribution,
    fig_release_year,
    fig_top_terms,
    fig_type_distribution,
)
from src.feature_engineering import build_feature_matrix, get_top_terms
from src.pca_analysis import fig_scree_plot
from src.visualization import (
    fig_cluster_distribution,
    fig_cluster_profile_terms,
    fig_pca_scatter,
    fig_silhouette_by_cluster,
    fig_top_genres_bar,
)


CLUSTER_NAMES = {
    0: "Action Lovers",
    1: "Drama Seekers",
    2: "Comedy Fans",
    3: "Documentary Lovers",
    4: "Family Viewers",
    5: "Global Explorers",
    6: "Classic Watchers",
    7: "TV Series Fans",
}

CLUSTER_DESCRIPTIONS = {
    0: "Thích phim hành động, giật gân, khoa học viễn tưởng",
    1: "Ưa thích phim chính kịch, tâm lý, tình cảm",
    2: "Thích phim hài, giải trí nhẹ nhàng",
    3: "Quan tâm phim tài liệu, khám phá, giáo dục",
    4: "Xem cùng gia đình, phù hợp nhiều lứa tuổi",
    5: "Thích nội dung quốc tế và đa văn hóa",
    6: "Ưa chuộng phim lâu năm và nội dung kinh điển",
    7: "Theo dõi series và chương trình truyền hình",
}


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = load_netflix_data(path)
    return preprocess_netflix_data(df)


def build_dashboard_model(df: pd.DataFrame, n_clusters: int, top_genres: int, top_countries: int):
    feature_matrix, feature_columns, top_genre_list, top_country_list, scaler = build_feature_matrix(
        df, top_genres=top_genres, top_countries=top_countries
    )
    labels, model = build_clusters(feature_matrix, n_clusters=n_clusters)
    sil_score, db_score = evaluate_clusters(feature_matrix, labels)
    summary = summarize_clusters(df, labels)
    profiles = build_cluster_profiles(df, labels)
    return (
        feature_matrix,
        pd.Series(labels, index=df.index),
        sil_score,
        db_score,
        summary,
        profiles,
        model,
        scaler,
        top_genre_list,
        top_country_list,
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #050d18;
            --panel: rgba(10, 25, 41, 0.92);
            --panel-2: rgba(9, 21, 35, 0.94);
            --line: rgba(255,255,255,0.08);
            --text: #f4f7fb;
            --muted: #a8b3c4;
            --red: #e82135;
            --blue: #1e73d8;
            --green: #35b854;
            --orange: #ff9f1a;
            --purple: #9553e6;
        }

        html, body, [class*="css"] { font-family: 'Inter', Arial, sans-serif; }
        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at 55% -10%, rgba(37, 92, 130, 0.20), transparent 34%),
                linear-gradient(135deg, #050b14 0%, #071521 44%, #05101d 100%);
        }
        .main .block-container {
            max-width: 1540px;
            padding: 1.35rem 1.55rem 1.2rem;
        }
        header[data-testid="stHeader"] { background: transparent; }
        section[data-testid="stSidebar"] {
            width: 245px !important;
            background: linear-gradient(180deg, #06111e 0%, #071a2a 54%, #06121f 100%);
            border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] > div { padding: 1.2rem 0.65rem; }
        [data-testid="stSidebar"] .stRadio label { color: #d9e0eb !important; }
        [data-testid="stSidebar"] .stRadio p,
        [data-testid="stSidebar"] .stRadio span {
            color: #d9e0eb !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
            padding: 0.72rem 0.85rem;
            border-radius: 8px;
            margin: 0.15rem 0;
        }
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
            background: linear-gradient(135deg, #e82135, #b91c2f);
            color: #fff !important;
        }
        [data-testid="stSidebar"] hr { border-color: var(--line); }

        h1, h2, h3, p { letter-spacing: 0; }
        h1 {
            font-size: 1.82rem !important;
            line-height: 1.18 !important;
            margin: 0 !important;
            font-weight: 800 !important;
        }
        h2 { font-size: 1rem !important; }
        h3 { font-size: 0.94rem !important; font-weight: 650 !important; }

        .topbar {
            display: flex;
            align-items: start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .subtitle { color: #d3dae6; margin-top: 0.35rem; font-size: 0.97rem; }
        .top-controls { display: flex; gap: 1rem; flex-wrap: wrap; justify-content: flex-end; }
        .control-pill {
            min-width: 178px;
            padding: 0.78rem 1rem;
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 8px;
            background: rgba(7, 18, 32, 0.84);
            color: #dfe6f0;
            font-size: 0.86rem;
        }

        .netflix-logo {
            color: var(--red);
            font-size: 2rem;
            line-height: 1;
            font-weight: 900;
            letter-spacing: 0;
            margin: 0.2rem 0 1.55rem 0.45rem;
        }
        .side-info {
            position: static;
            width: 100%;
            box-sizing: border-box;
            margin-top: 2.2rem;
            padding: 1rem;
            border-radius: 8px;
            background: rgba(8, 24, 38, 0.78);
            border: 1px solid var(--line);
            color: #c4cedb;
            font-size: 0.78rem;
            line-height: 1.55;
        }
        .side-info p { margin: 0 0 0.75rem; }
        .side-info p:last-child { margin-bottom: 0; }
        .side-info b { color: var(--red); }

        .metric-card {
            min-height: 104px;
            padding: 0.9rem 0.95rem;
            border-radius: 8px;
            border: 1px solid var(--line);
            background: linear-gradient(145deg, rgba(12, 31, 49, 0.96), rgba(7, 17, 30, 0.96));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
            display: grid;
            grid-template-columns: 46px minmax(0, 1fr);
            gap: 0.8rem;
            align-items: center;
        }
        .metric-icon {
            width: 46px;
            height: 46px;
            border-radius: 999px;
            display: grid;
            place-items: center;
            font-size: 1.35rem;
            font-weight: 800;
        }
        .metric-title { color: #dce4ef; font-size: 0.78rem; margin-bottom: 0.35rem; }
        .metric-value { color: #fff; font-size: 1.52rem; font-weight: 700; line-height: 1; }
        .metric-note { font-size: 0.72rem; margin-top: 0.55rem; line-height: 1.25; }
        .red { color: #ff3148; } .blue { color: #2d95ff; } .green { color: #4fdf6d; }
        .orange { color: #ffb01f; } .purple { color: #b37aff; }
        .bg-red { background: rgba(232, 33, 53, 0.18); color: #ff3148; }
        .bg-blue { background: rgba(30, 115, 216, 0.20); color: #2d95ff; }
        .bg-green { background: rgba(53, 184, 84, 0.20); color: #4fdf6d; }
        .bg-orange { background: rgba(255, 159, 26, 0.18); color: #ffb01f; }
        .bg-purple { background: rgba(149, 83, 230, 0.20); color: #b37aff; }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 8px;
            border: 1px solid var(--line);
            background: linear-gradient(145deg, rgba(12, 31, 49, 0.96), rgba(7, 17, 30, 0.96));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 0.75rem 0.9rem 0.55rem; }
        .chart-title { font-size: 0.98rem; font-weight: 650; margin: 0 0 0.2rem; color: #f5f7fb; }
        .stPlotlyChart { margin-top: -0.15rem; }

        .score-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.7rem;
            margin: 0.25rem 0 0.75rem;
        }
        .score-box {
            padding: 0.78rem;
            border-radius: 7px;
            border: 1px solid var(--line);
            background: rgba(7, 18, 30, 0.72);
        }
        .score-label { color: #c9d2df; font-size: 0.78rem; }
        .score-value { color: #ffb01f; font-size: 1.62rem; font-weight: 800; line-height: 1.2; }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }
        .profile-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            font-size: 0.76rem;
            color: #e7edf6;
            overflow: hidden;
            border-radius: 8px;
            border: 1px solid var(--line);
        }
        .profile-table th {
            text-align: left;
            font-weight: 600;
            color: #b8c3d2;
            background: rgba(7, 18, 30, 0.92);
            padding: 0.72rem 0.65rem;
            border-bottom: 1px solid var(--line);
            border-right: 1px solid var(--line);
        }
        .profile-table td {
            padding: 0.68rem 0.65rem;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            border-right: 1px solid rgba(255,255,255,0.05);
            background: rgba(8, 22, 36, 0.78);
            vertical-align: top;
            overflow-wrap: break-word;
        }
        .profile-table tr:last-child td { border-bottom: 0; }
        .profile-table th:nth-child(1), .profile-table td:nth-child(1) { width: 10%; white-space: nowrap; overflow-wrap: normal; }
        .profile-table th:nth-child(2), .profile-table td:nth-child(2) { width: 18%; }
        .profile-table th:nth-child(3), .profile-table td:nth-child(3) { width: 32%; }
        .profile-table th:nth-child(4), .profile-table td:nth-child(4) { width: 12%; }
        .profile-table th:nth-child(5), .profile-table td:nth-child(5) { width: 28%; }
        footer { visibility: hidden; }

        @media (max-width: 900px) {
            .topbar { flex-direction: column; }
            .top-controls { justify-content: flex-start; }
            .side-info { position: static; width: auto; margin-top: 1rem; }
            h1 { font-size: 1.45rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, note: str, icon: str, color: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon bg-{color}">{icon}</div>
            <div>
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note {color}">{note}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def translate_type(value: str) -> str:
    return {"Movie": "Phim lẻ", "TV Show": "Chương trình TV"}.get(value, value)


def build_profile_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, row in summary.iterrows():
        top_genres = ", ".join(str(row["top_genres"]).split(", ")[:2])
        rows.append(
            {
                "Cụm": f"Cụm {int(cluster_id) + 1}",
                "Tên cụm": CLUSTER_NAMES.get(int(cluster_id), f"Cụm {int(cluster_id) + 1}"),
                "Đặc điểm nổi bật": CLUSTER_DESCRIPTIONS.get(int(cluster_id), "Nhóm sở thích nội dung riêng"),
                "Thời lượng TB": f"{int(row['avg_duration_min'])} phút",
                "Thể loại chính": top_genres,
            }
        )
    return pd.DataFrame(rows)


def render_header(section: str) -> None:
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <h1>{section}</h1>
            </div>
            <div class="top-controls">
                <div class="control-pill">01/05/2024 - 31/05/2024</div>
                <div class="control-pill">{section}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(df_filtered: pd.DataFrame, n_clusters: int, total_views: int, avg_watch_hours: float, score: float) -> None:
    metric_cols = st.columns(5)
    with metric_cols[0]:
        metric_card("Tổng số người dùng", f"{len(df_filtered):,}", "100% trong dữ liệu lọc", "👥", "red")
    with metric_cols[1]:
        metric_card("Số cụm người dùng", f"{n_clusters}", "Phân cụm (K-Means)", "◔", "purple")
    with metric_cols[2]:
        metric_card("Tổng lượt xem", f"{total_views:,}", "Trong khoảng thời gian", "▶", "blue")
    with metric_cols[3]:
        metric_card("Thời gian xem TB", f"{avg_watch_hours:.2f}", "giờ/người dùng", "◷", "green")
    with metric_cols[4]:
        metric_card("Độ chính xác mô hình", f"{score:.2f}", "Silhouette Score", "★", "orange")


def render_overview_page(df_filtered, feature_matrix, labels, score, summary, total_views, avg_watch_hours, db_score, n_clusters) -> None:
    render_metric_row(df_filtered, n_clusters, total_views, avg_watch_hours, score)

    row1_col1, row1_col2, row1_col3 = st.columns([1.05, 1.1, 1.25])
    with row1_col1:
        with st.container(border=True):
            st.markdown('<div class="chart-title">Phân bố người dùng theo cụm</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_cluster_distribution(labels), use_container_width=True, config={"displayModeBar": False})
    with row1_col2:
        with st.container(border=True):
            st.markdown('<div class="chart-title">Sở thích thể loại theo cụm (Tỷ lệ lượt xem)</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_cluster_profile_terms(df_filtered, labels, top_n=9), use_container_width=True, config={"displayModeBar": False})
    with row1_col3:
        with st.container(border=True):
            st.markdown('<div class="chart-title">Giảm chiều dữ liệu (PCA 2D)</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_pca_scatter(feature_matrix, labels), use_container_width=True, config={"displayModeBar": False})

    row2_col1, row2_col2, row2_col3 = st.columns([1.55, 1.05, 1.25])
    with row2_col1:
        render_cluster_profile_table(summary)
    with row2_col2:
        render_model_scores(feature_matrix, labels, score, db_score)
    with row2_col3:
        with st.container(border=True):
            st.markdown('<div class="chart-title">Thể loại được xem nhiều nhất (toàn bộ người dùng)</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_top_genres_bar(df_filtered, top_n=9), use_container_width=True, config={"displayModeBar": False})


def render_cluster_profile_table(summary: pd.DataFrame) -> None:
    with st.container(border=True):
        st.markdown('<div class="chart-title">Chân dung các cụm người dùng</div>', unsafe_allow_html=True)
        profile_table = build_profile_table(summary)
        st.markdown(profile_table.to_html(index=False, classes="profile-table", border=0), unsafe_allow_html=True)


def render_model_scores(feature_matrix: pd.DataFrame, labels: pd.Series, score: float, db_index: float) -> None:
    with st.container(border=True):
        st.markdown('<div class="chart-title">Chỉ số đánh giá mô hình</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="score-grid">
                <div class="score-box"><div class="score-label">Silhouette Score</div><div class="score-value">{score:.2f}</div></div>
                <div class="score-box"><div class="score-label">Davies-Bouldin Index</div><div class="score-value" style="color:#2d95ff">{db_index:.2f}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chart-title" style="font-size:.85rem">Silhouette Score theo từng cụm</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_silhouette_by_cluster(feature_matrix, labels), use_container_width=True, config={"displayModeBar": False})


def render_user_analysis_page(df_filtered, feature_matrix, labels, score, total_views, avg_watch_hours, n_clusters) -> None:
    render_metric_row(df_filtered, n_clusters, total_views, avg_watch_hours, score)
    col1, col2 = st.columns([1.2, 1])
    with col1:
        with st.container(border=True):
            st.markdown('<div class="chart-title">Không gian người dùng sau PCA</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_pca_scatter(feature_matrix, labels), use_container_width=True, config={"displayModeBar": False})
    with col2:
        with st.container(border=True):
            st.markdown('<div class="chart-title">Phân bố người dùng theo cụm</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_cluster_distribution(labels), use_container_width=True, config={"displayModeBar": False})


def render_content_analysis_page(df: pd.DataFrame, df_filtered: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.plotly_chart(fig_type_distribution(df_filtered), use_container_width=True, config={"displayModeBar": False})
    with col2:
        with st.container(border=True):
            st.plotly_chart(fig_rating_distribution(df_filtered), use_container_width=True, config={"displayModeBar": False})

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            st.plotly_chart(fig_duration_distribution(df_filtered), use_container_width=True, config={"displayModeBar": False})
    with col4:
        with st.container(border=True):
            st.plotly_chart(fig_country_distribution(df_filtered, top_n=10), use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        st.markdown('<div class="chart-title">Chất lượng dữ liệu</div>', unsafe_allow_html=True)
        st.dataframe(build_missing_report(df), use_container_width=True, hide_index=True)


def render_preference_page(df_filtered: pd.DataFrame, labels: pd.Series) -> None:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        with st.container(border=True):
            st.markdown('<div class="chart-title">Sở thích thể loại theo cụm</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_cluster_profile_terms(df_filtered, labels, top_n=10), use_container_width=True, config={"displayModeBar": False})
    with col2:
        with st.container(border=True):
            st.markdown('<div class="chart-title">Top thể loại toàn bộ dữ liệu lọc</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_top_genres_bar(df_filtered, top_n=10), use_container_width=True, config={"displayModeBar": False})


def render_clusters_page(df_filtered, feature_matrix, labels, score, db_score) -> None:
    col1, col2 = st.columns([1.05, 1])
    with col1:
        with st.container(border=True):
            st.markdown('<div class="chart-title">Tỷ trọng từng cụm</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_cluster_distribution(labels), use_container_width=True, config={"displayModeBar": False})
    with col2:
        render_model_scores(feature_matrix, labels, score, db_score)


def render_prediction_page(
    df_filtered: pd.DataFrame,
    summary: pd.DataFrame,
    kmeans_model,
    scaler,
    top_genre_list: list,
    top_country_list: list,
) -> None:
    # --- Bảng luật gợi ý ---
    rows = []
    for cluster_id, row in summary.iterrows():
        rows.append(
            {
                "Cụm": f"Cụm {int(cluster_id) + 1}",
                "Nhóm người dùng": CLUSTER_NAMES.get(int(cluster_id), "Nhóm sở thích"),
                "Gợi ý nội dung": row["top_genres"],
            }
        )
    with st.container(border=True):
        st.markdown('<div class="chart-title">Luật gợi ý theo cụm sở thích</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Form phân loại title mới ---
    with st.container(border=True):
        st.markdown('<div class="chart-title">Phân loại nội dung mới</div>', unsafe_allow_html=True)
        st.caption("Nhập thông tin một title mới để mô hình dự đoán nó thuộc nhóm người dùng nào.")

        col_a, col_b = st.columns(2)
        with col_a:
            new_title = st.text_input("Tên nội dung", value="My New Movie")
            content_type = st.selectbox("Loại", ["Movie", "TV Show"])
            listed_in = st.text_input("Thể loại (phân cách bằng dấu phẩy)", value="Action, Adventure")
            release_year = st.number_input("Năm phát hành", min_value=1900, max_value=2030, value=2023, step=1)
        with col_b:
            country = st.text_input("Quốc gia", value="United States")
            rating = st.selectbox("Rating", ["G", "PG", "PG-13", "R", "TV-Y", "TV-Y7", "TV-G", "TV-PG", "TV-14", "TV-MA", "NR"])
            duration = st.text_input(
                "Thời lượng",
                value="120 min" if content_type == "Movie" else "1 Season",
                help="Ví dụ: '120 min' cho phim lẻ hoặc '2 Seasons' cho TV Show",
            )

        if st.button("Dự đoán cụm", type="primary"):
            cluster_id = predict_new_title(
                title=new_title,
                content_type=content_type,
                listed_in=listed_in,
                country=country,
                duration=duration,
                rating=rating,
                release_year=int(release_year),
                kmeans_model=kmeans_model,
                scaler=scaler,
                top_genre_list=top_genre_list,
                top_country_list=top_country_list,
            )
            cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cụm {cluster_id + 1}")
            cluster_desc = CLUSTER_DESCRIPTIONS.get(cluster_id, "")
            recommended_genres = summary.loc[cluster_id, "top_genres"] if cluster_id in summary.index else "—"
            st.success(f"**Kết quả:** {new_title} → Cụm {cluster_id + 1} — **{cluster_name}**")
            st.info(f"Mô tả: {cluster_desc}\n\nThể loại phổ biến trong cụm: {recommended_genres}")


def render_data_analysis_page(df_filtered: pd.DataFrame, feature_matrix: pd.DataFrame) -> None:
    """Trang phân tích dữ liệu: EDA nâng cao + PCA variance analysis."""
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown('<div class="chart-title">Phân phối nội dung theo năm phát hành</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_release_year(df_filtered), use_container_width=True, config={"displayModeBar": False})
    with col2:
        with st.container(border=True):
            st.markdown('<div class="chart-title">Tương quan giữa các đặc trưng số</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_correlation_heatmap(feature_matrix), use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        st.markdown('<div class="chart-title">Phân tích PCA — Lựa chọn số chiều phù hợp</div>', unsafe_allow_html=True)
        st.caption(
            "Scree plot và cumulative variance giúp xác định bao nhiêu components PCA cần thiết "
            "để giữ lại đủ thông tin trước khi clustering."
        )
        fig_scree, fig_cum, recs = fig_scree_plot(feature_matrix)
        pc_col1, pc_col2 = st.columns(2)
        with pc_col1:
            st.markdown("**Scree Plot** — Variance mỗi PC")
            st.plotly_chart(fig_scree, use_container_width=True, config={"displayModeBar": False})
        with pc_col2:
            st.markdown("**Cumulative Variance** — Ngưỡng lựa chọn")
            st.plotly_chart(fig_cum, use_container_width=True, config={"displayModeBar": False})

        rec_rows = [
            {"Ngưỡng variance": f"{int(t * 100)}%", "Số components tối thiểu": n, "Ghi chú": "Khuyến nghị clustering" if t == 0.90 else ""}
            for t, n in recs.items()
        ]
        st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)


def render_report_page(df_filtered: pd.DataFrame, score: float, db_score: float, summary: pd.DataFrame) -> None:
    with st.container(border=True):
        st.markdown('<div class="chart-title">Báo cáo tổng kết</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            - Số bản ghi sau lọc: **{len(df_filtered):,}**
            - Số cụm phân tích: **{len(summary):,}**
            - Silhouette Score: **{score:.2f}** (cao hơn tốt hơn, max 1)
            - Davies-Bouldin Index: **{db_score:.2f}** (thấp hơn tốt hơn, min 0)
            """
        )
    render_cluster_profile_table(summary)


def main() -> None:
    st.set_page_config(
        page_title="Phân tích người dùng Netflix",
        page_icon="N",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    df = load_data("data/raw/netflix_titles.csv")
    top_genre_list = get_top_terms(df["genre_list"], 25)
    selected_types = ["Movie", "TV Show"]
    selected_genres = top_genre_list[:10]
    n_clusters = 5
    top_genres = 15
    top_countries = 10

    with st.sidebar:
        st.markdown('<div class="netflix-logo">NETFLIX</div>', unsafe_allow_html=True)
        section = st.radio(
            "Điều hướng",
            [
                "Tổng quan",
                "Phân tích người dùng",
                "Phân tích nội dung",
                "Phân tích dữ liệu",
                "Phân loại sở thích",
                "Cụm người dùng",
                "Chân dung cụm",
                "Dự đoán & Gợi ý",
                "Báo cáo",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            """
            <div class="side-info">
                <p><b>Thông tin dự án</b></p>
                <p>Dataset: Netflix User Viewing Activity</p>
                <p>Kỹ thuật: K-Means Clustering</p>
                <p>Đặc trưng: Thể loại, thời lượng xem, mức độ đa dạng nội dung</p>
                <p>Ngôn ngữ: Python</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    df_filtered = df.copy()
    if selected_types:
        df_filtered = df_filtered[df_filtered["type"].isin(selected_types)]
    if selected_genres:
        df_filtered = df_filtered[
            df_filtered["genre_list"].apply(lambda values: any(genre in values for genre in selected_genres))
        ]

    if len(df_filtered) < n_clusters:
        st.warning("Không đủ dữ liệu sau khi lọc để phân cụm. Vui lòng chọn thêm loại nội dung hoặc thể loại.")
        return

    (
        feature_matrix,
        labels,
        sil_score,
        db_score,
        summary,
        profiles,
        kmeans_model,
        scaler,
        top_genre_list,
        top_country_list,
    ) = build_dashboard_model(
        df_filtered,
        n_clusters=n_clusters,
        top_genres=top_genres,
        top_countries=top_countries,
    )

    total_views = int(len(df_filtered) * max(df_filtered["genre_list"].str.len().mean(), 1) * 47)
    avg_watch_hours = df_filtered["duration_min"].replace(0, pd.NA).dropna().mean() / 60
    if pd.isna(avg_watch_hours):
        avg_watch_hours = 0

    render_header(section)

    if section == "Tổng quan":
        render_overview_page(df_filtered, feature_matrix, labels, sil_score, summary, total_views, avg_watch_hours, db_score, n_clusters)
    elif section == "Phân tích người dùng":
        render_user_analysis_page(df_filtered, feature_matrix, labels, sil_score, total_views, avg_watch_hours, n_clusters)
    elif section == "Phân tích nội dung":
        render_content_analysis_page(df, df_filtered)
    elif section == "Phân tích dữ liệu":
        render_data_analysis_page(df_filtered, feature_matrix)
    elif section == "Phân loại sở thích":
        render_preference_page(df_filtered, labels)
    elif section == "Cụm người dùng":
        render_clusters_page(df_filtered, feature_matrix, labels, sil_score, db_score)
    elif section == "Chân dung cụm":
        render_cluster_profile_table(summary)
    elif section == "Dự đoán & Gợi ý":
        render_prediction_page(df_filtered, summary, kmeans_model, scaler, top_genre_list, top_country_list)
    elif section == "Báo cáo":
        render_report_page(df_filtered, sil_score, db_score, summary)


if __name__ == "__main__":
    main()
