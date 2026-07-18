"""
Export tất cả biểu đồ thực ra PNG để dùng trong PPTX.
Chạy: python export_charts.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

OUT = Path("image/charts")
OUT.mkdir(parents=True, exist_ok=True)

BG_DARK  = "#0d1b2a"
BG_PANEL = "#0a1828"
FONT_CLR = "#f0f4fc"
MUTED    = "#8899aa"
GRID     = "rgba(200,220,255,0.07)"
COLORS   = ["#e82135","#1976d2","#43a047","#8e44cc","#fb8c00","#00acc1","#fdd835","#ec407a"]

def base_layout(fig, h=480, showlegend=False, margin=None):
    fig.update_layout(
        height=h,
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PANEL,
        font=dict(color=FONT_CLR, family="Arial, sans-serif", size=13),
        margin=margin or dict(l=30, r=30, t=46, b=40),
        showlegend=showlegend,
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=FONT_CLR, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=FONT_CLR, linecolor=GRID),
        legend=dict(bgcolor="rgba(8,22,37,0.7)", bordercolor="rgba(255,255,255,0.08)",
                    borderwidth=1, font=dict(color=FONT_CLR, size=12)),
    )
    return fig

# ─── Load & Preprocess ────────────────────────────────────────────────────
print("Loading data...")
df_raw = pd.read_csv("data/raw/netflix_titles.csv")
df = df_raw.copy()
df = df.drop_duplicates(subset=["show_id"])
df["type"]     = df["type"].fillna("Unknown")
df["country"]  = df["country"].fillna("Unknown")
df["listed_in"]= df["listed_in"].fillna("Unknown")
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)

def parse_dur(d):
    if pd.isna(d): return {"duration_min":0,"season_count":0}
    v = str(d).strip()
    if "min" in v:
        try: return {"duration_min":int(v.replace("min","").strip()),"season_count":0}
        except: return {"duration_min":0,"season_count":0}
    if "Season" in v:
        try: return {"duration_min":0,"season_count":int(v.split()[0])}
        except: return {"duration_min":0,"season_count":0}
    return {"duration_min":0,"season_count":0}

dur = df["duration"].apply(parse_dur).apply(pd.Series)
df  = pd.concat([df, dur], axis=1)
df["duration_min"]  = df["duration_min"].fillna(0).astype(int)
df["season_count"]  = df["season_count"].fillna(0).astype(int)

rating_map = {"G":1,"PG":2,"PG-13":3,"TV-Y":1,"TV-Y7":1,"TV-G":1,"TV-PG":2,"TV-14":3,"TV-MA":4,"R":4,"NR":0,"UR":0,"Unknown":0}
df["rating"]       = df["rating"].fillna("Unknown")
df["rating_level"] = df["rating"].map(rating_map).fillna(0).astype(int)
df["genre_list"]   = df["listed_in"].astype(str).apply(lambda t: [g.strip() for g in t.split(",") if g.strip()])
df["country_list"] = df["country"].astype(str).apply(lambda t: [c.strip() for c in t.split(",") if c.strip()])

# Feature matrix
def get_top(series, n):
    cnt={}
    for vals in series:
        for v in vals: cnt[v]=cnt.get(v,0)+1
    return [k for k,_ in sorted(cnt.items(),key=lambda x:-x[1])[:n]]

TOP_GENRES    = get_top(df["genre_list"], 15)
TOP_COUNTRIES = get_top(df["country_list"], 10)

feat = pd.DataFrame(index=df.index)
feat["is_movie"]  = (df["type"]=="Movie").astype(int)
feat["is_tvshow"] = (df["type"]=="TV Show").astype(int)
for g in TOP_GENRES:
    feat[f"genre_{g}"] = df["genre_list"].apply(lambda v: int(g in v))
for c in TOP_COUNTRIES:
    feat[f"country_{c}"] = df["country_list"].apply(lambda v: int(c in v))
feat["dur_s"]    = df["duration_min"].astype(float)
feat["season_s"] = df["season_count"].astype(float)
feat["rating_s"] = df["rating_level"].astype(float)
scaler = MinMaxScaler()
feat[["dur_s","season_s","rating_s"]] = scaler.fit_transform(feat[["dur_s","season_s","rating_s"]])

# K-Means K=5
print("Running KMeans K=5...")
km = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = km.fit_predict(feat)
df["cluster"] = labels

CLUSTER_NAMES = {0:"Action Lovers",1:"Drama Seekers",2:"Comedy Fans",3:"Documentary Lovers",4:"Family Viewers"}

def save(fig, name, w=1280, h=None):
    path = OUT / f"{name}.png"
    kwargs = dict(format="png", width=w, engine="kaleido")
    if h: kwargs["height"]=h
    fig.write_image(str(path), **kwargs)
    print(f"  Saved {path}")

# ═══════════════════════════════════════════════════════════════════════
# 1. TYPE DISTRIBUTION (Pie)
# ═══════════════════════════════════════════════════════════════════════
print("Chart 1: Type Distribution...")
type_cnt = df["type"].value_counts()
fig = go.Figure(go.Pie(
    labels=type_cnt.index.tolist(),
    values=type_cnt.values.tolist(),
    hole=0.52,
    marker=dict(colors=["#e82135","#1976d2"], line=dict(color="#07111f",width=3)),
    textinfo="percent+label",
    textfont=dict(size=15, color="#fff"),
    hovertemplate="%{label}: %{value:,}<extra></extra>",
))
fig.update_layout(
    title=dict(text="Phân phối loại nội dung (Movie vs TV Show)", font=dict(size=18,color=FONT_CLR), x=0.5),
    annotations=[dict(text=f"<b>{len(df):,}</b><br>Titles", x=0.5, y=0.5,
                      showarrow=False, font=dict(size=16,color=FONT_CLR))],
    height=480, paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
    font=dict(color=FONT_CLR, family="Arial"),
    margin=dict(l=30,r=30,t=60,b=30), showlegend=True,
    legend=dict(bgcolor="rgba(8,22,37,0.7)",bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1,font=dict(color=FONT_CLR,size=13)),
)
save(fig, "01_type_distribution", h=480)

# ═══════════════════════════════════════════════════════════════════════
# 2. TOP GENRES BAR
# ═══════════════════════════════════════════════════════════════════════
print("Chart 2: Top Genres...")
genre_cnt = df["genre_list"].explode().value_counts().head(12).sort_values()
fig = go.Figure(go.Bar(
    x=genre_cnt.values, y=genre_cnt.index,
    orientation="h",
    marker=dict(color=genre_cnt.values, colorscale="Blues",
                line=dict(color="rgba(255,255,255,0.1)", width=1)),
    text=[f"{v:,}" for v in genre_cnt.values],
    textposition="outside", cliponaxis=False,
    hovertemplate="%{y}: %{x:,}<extra></extra>",
))
fig.update_layout(
    title=dict(text="Top 12 thể loại nội dung phổ biến nhất", font=dict(size=18,color=FONT_CLR), x=0.5),
    height=500, paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
    font=dict(color=FONT_CLR, family="Arial"),
    margin=dict(l=170,r=80,t=60,b=40),
    xaxis=dict(gridcolor=GRID, color=FONT_CLR, title="Số lượng tiêu đề"),
    yaxis=dict(gridcolor=GRID, color=FONT_CLR, tickfont=dict(size=12)),
    showlegend=False,
)
save(fig, "02_top_genres", h=500)

# ═══════════════════════════════════════════════════════════════════════
# 3. DURATION DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════
print("Chart 3: Duration Distribution...")
movies = df[df["type"]=="Movie"]["duration_min"].replace(0,np.nan).dropna()
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=movies, nbinsx=35, name="Phim lẻ",
    marker=dict(color="#e82135", opacity=0.85, line=dict(color="#ff5566",width=0.5)),
    hovertemplate="Khoảng %{x} phút: %{y} phim<extra></extra>",
))
fig.update_layout(
    title=dict(text="Phân phối thời lượng phim lẻ (phút)", font=dict(size=18,color=FONT_CLR), x=0.5),
    height=440, paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
    font=dict(color=FONT_CLR, family="Arial"),
    margin=dict(l=50,r=30,t=60,b=50),
    xaxis=dict(gridcolor=GRID, color=FONT_CLR, title="Thời lượng (phút)", range=[0,250]),
    yaxis=dict(gridcolor=GRID, color=FONT_CLR, title="Số lượng phim"),
    showlegend=False,
    bargap=0.05,
)
save(fig, "03_duration_dist", h=440)

# ═══════════════════════════════════════════════════════════════════════
# 4. RATING DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════
print("Chart 4: Rating Distribution...")
rating_cnt = df["rating"].value_counts().head(10).sort_values()
colors_rating = px.colors.sequential.Reds[2:]
fig = go.Figure(go.Bar(
    x=rating_cnt.values, y=rating_cnt.index,
    orientation="h",
    marker=dict(color=rating_cnt.values, colorscale="Reds",
                line=dict(color="rgba(255,255,255,0.08)",width=0.5)),
    text=[f"{v:,}" for v in rating_cnt.values],
    textposition="outside", cliponaxis=False,
    hovertemplate="%{y}: %{x:,}<extra></extra>",
))
fig.update_layout(
    title=dict(text="Phân phối xếp hạng độ tuổi (Rating)", font=dict(size=18,color=FONT_CLR), x=0.5),
    height=440, paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
    font=dict(color=FONT_CLR, family="Arial"),
    margin=dict(l=100,r=80,t=60,b=40),
    xaxis=dict(gridcolor=GRID, color=FONT_CLR, title="Số lượng"),
    yaxis=dict(gridcolor=GRID, color=FONT_CLR, tickfont=dict(size=12)),
    showlegend=False,
)
save(fig, "04_rating_dist", h=440)

# ═══════════════════════════════════════════════════════════════════════
# 5. CORRELATION HEATMAP (thực)
# ═══════════════════════════════════════════════════════════════════════
print("Chart 5: Correlation Heatmap...")
rfm_cols = ["dur_s","season_s","rating_s"]
rfm_labels = ["Duration\n(scaled)","Season\n(scaled)","Rating\n(scaled)"]
numeric_feat = feat[rfm_cols].copy()

# Thêm genre features đầu tiên
genre_cols = [f"genre_{g}" for g in TOP_GENRES[:5]]
for gc in genre_cols:
    numeric_feat[gc] = feat[gc]

all_cols = rfm_cols + genre_cols
all_labels = ["Duration","Season","Rating"] + [g.replace("International Movies","Intl. Movies")
              .replace("International TV Shows","Intl. TV Shows")[:14] for g in TOP_GENRES[:5]]

corr = numeric_feat[all_cols].corr().round(2)
fig = go.Figure(go.Heatmap(
    z=corr.values,
    x=all_labels,
    y=all_labels,
    colorscale=[
        [0.0, "#1a3a6b"], [0.4, "#0d4fa0"], [0.5, "#0d1b2a"],
        [0.6, "#8b1a1a"], [0.8, "#c0392b"], [1.0, "#e82135"]
    ],
    zmid=0, zmin=-1, zmax=1,
    text=corr.values.round(2),
    texttemplate="%{text}",
    textfont=dict(size=12, color="white"),
    hovertemplate="%{x} × %{y}: %{z:.2f}<extra></extra>",
    colorbar=dict(title=dict(text="r", font=dict(color=FONT_CLR)), tickfont=dict(color=FONT_CLR)),
))
fig.update_layout(
    title=dict(text="Ma trận tương quan giữa các đặc trưng", font=dict(size=18,color=FONT_CLR), x=0.5),
    height=520, paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
    font=dict(color=FONT_CLR, family="Arial"),
    margin=dict(l=140,r=30,t=70,b=120),
    xaxis=dict(color=FONT_CLR, tickangle=-35, tickfont=dict(size=11)),
    yaxis=dict(color=FONT_CLR, tickfont=dict(size=11)),
)
save(fig, "05_correlation_heatmap", h=520)

# ═══════════════════════════════════════════════════════════════════════
# 6. PCA SCATTER
# ═══════════════════════════════════════════════════════════════════════
print("Chart 6: PCA Scatter...")
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(feat)
exp = pca.explained_variance_ratio_ * 100

scatter_df = pd.DataFrame({"pc1":pcs[:,0],"pc2":pcs[:,1],"cluster":labels})
fig = go.Figure()
for cid in sorted(scatter_df["cluster"].unique()):
    sub = scatter_df[scatter_df["cluster"]==cid]
    fig.add_trace(go.Scattergl(
        x=sub["pc1"], y=sub["pc2"], mode="markers",
        name=f"Cụm {cid+1} – {CLUSTER_NAMES.get(cid,'')}",
        marker=dict(color=COLORS[cid%len(COLORS)], size=5, opacity=0.75,
                    line=dict(width=0)),
        hovertemplate=f"Cụm {cid+1}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>",
    ))
fig.update_layout(
    title=dict(text=f"PCA 2D – Phân bố các cụm nội dung<br><sup>PC1={exp[0]:.1f}%  PC2={exp[1]:.1f}%  variance explained</sup>",
               font=dict(size=17,color=FONT_CLR), x=0.5),
    height=520, paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
    font=dict(color=FONT_CLR, family="Arial"),
    margin=dict(l=50,r=30,t=80,b=50),
    xaxis=dict(gridcolor=GRID, color=FONT_CLR, zerolinecolor=GRID, title=f"PC1 ({exp[0]:.1f}%)"),
    yaxis=dict(gridcolor=GRID, color=FONT_CLR, zerolinecolor=GRID, title=f"PC2 ({exp[1]:.1f}%)"),
    showlegend=True,
    legend=dict(bgcolor="rgba(8,22,37,0.85)", bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1, font=dict(color=FONT_CLR,size=12),
                x=0.01, y=0.99, xanchor="left", yanchor="top"),
)
save(fig, "06_pca_scatter", h=520)

# ═══════════════════════════════════════════════════════════════════════
# 7. ELBOW
# ═══════════════════════════════════════════════════════════════════════
print("Chart 7: Elbow...")
inertias = []
K_range = range(2, 11)
for k in K_range:
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_tmp.fit(feat)
    inertias.append(km_tmp.inertia_)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(K_range), y=inertias, mode="lines+markers",
    line=dict(color="#1976d2", width=3),
    marker=dict(color=COLORS[:len(K_range)], size=10,
                line=dict(color="#fff",width=1.5)),
    name="WCSS / Inertia",
    hovertemplate="K=%{x}<br>WCSS=%{y:,.0f}<extra></extra>",
))
# Highlight elbow K=5
elbow_idx = 3  # K=5 index in range(2,11) -> index 3
fig.add_trace(go.Scatter(
    x=[5], y=[inertias[3]],
    mode="markers+text",
    marker=dict(color="#e82135", size=18, symbol="circle",
                line=dict(color="#ff8888",width=2)),
    text=["K = 5 (Elbow)"], textposition="top center",
    textfont=dict(color="#e82135", size=13),
    showlegend=False,
    hovertemplate="Elbow tại K=5<br>WCSS=%{y:,.0f}<extra></extra>",
))
fig.update_layout(
    title=dict(text="Phương pháp Elbow – Chọn số cụm K tối ưu", font=dict(size=18,color=FONT_CLR), x=0.5),
    height=460, paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
    font=dict(color=FONT_CLR, family="Arial"),
    margin=dict(l=60,r=30,t=70,b=55),
    xaxis=dict(gridcolor=GRID, color=FONT_CLR, title="Số cụm K", tickmode="linear", tick0=2, dtick=1),
    yaxis=dict(gridcolor=GRID, color=FONT_CLR, title="WCSS (Inertia)"),
    showlegend=False,
)
save(fig, "07_elbow", h=460)

# ═══════════════════════════════════════════════════════════════════════
# 8. SILHOUETTE BY K
# ═══════════════════════════════════════════════════════════════════════
print("Chart 8: Silhouette by K...")
sil_scores_k = []
for k in K_range:
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl_tmp = km_tmp.fit_predict(feat)
    sc = silhouette_score(feat, lbl_tmp, sample_size=1500, random_state=42)
    sil_scores_k.append(round(sc, 4))

bar_colors = ["#e82135" if k==5 else "#1976d2" for k in K_range]
fig = go.Figure(go.Bar(
    x=list(K_range), y=sil_scores_k,
    marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.08)",width=0.5)),
    text=[f"{s:.3f}" for s in sil_scores_k],
    textposition="outside", cliponaxis=False,
    hovertemplate="K=%{x}<br>Silhouette=%{y:.4f}<extra></extra>",
))
fig.update_layout(
    title=dict(text="Silhouette Score theo số cụm K<br><sup>(Đỏ = K tối ưu được chọn)</sup>",
               font=dict(size=18,color=FONT_CLR), x=0.5),
    height=460, paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
    font=dict(color=FONT_CLR, family="Arial"),
    margin=dict(l=60,r=30,t=80,b=55),
    xaxis=dict(gridcolor=GRID, color=FONT_CLR, title="Số cụm K", tickmode="linear", tick0=2, dtick=1),
    yaxis=dict(gridcolor=GRID, color=FONT_CLR, title="Silhouette Score", range=[0, max(sil_scores_k)+0.08]),
    showlegend=False, bargap=0.35,
)
save(fig, "08_silhouette_k", h=460)
best_k  = list(K_range)[sil_scores_k.index(max(sil_scores_k))]
best_sc = max(sil_scores_k)
print(f"  Best K={best_k}, Silhouette={best_sc:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# 9. SILHOUETTE BY CLUSTER (K=5)
# ═══════════════════════════════════════════════════════════════════════
print("Chart 9: Silhouette by Cluster...")
samp = min(2000, len(feat))
feat_s = feat.sample(n=samp, random_state=42)
lbl_s  = pd.Series(labels).loc[feat_s.index]
sil_smp = silhouette_samples(feat_s, lbl_s)
sil_df  = pd.DataFrame({"cluster":lbl_s.values,"score":sil_smp})
sil_mean = sil_df.groupby("cluster")["score"].mean().sort_index()

fig = go.Figure(go.Bar(
    x=[f"Cụm {int(c)+1}\n{CLUSTER_NAMES.get(int(c),'')}" for c in sil_mean.index],
    y=sil_mean.values,
    marker=dict(color=[COLORS[int(c)%len(COLORS)] for c in sil_mean.index],
                line=dict(color="rgba(255,255,255,0.1)",width=0.5)),
    text=[f"{v:.3f}" for v in sil_mean.values],
    textposition="outside", cliponaxis=False,
    hovertemplate="%{x}: %{y:.4f}<extra></extra>",
))
fig.update_layout(
    title=dict(text=f"Silhouette Score từng cụm (K=5,  avg={sil_mean.mean():.3f})",
               font=dict(size=18,color=FONT_CLR), x=0.5),
    height=460, paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
    font=dict(color=FONT_CLR, family="Arial"),
    margin=dict(l=50,r=30,t=70,b=80),
    xaxis=dict(gridcolor=GRID, color=FONT_CLR, tickfont=dict(size=11)),
    yaxis=dict(gridcolor=GRID, color=FONT_CLR, title="Score", range=[0,max(sil_mean.values)+0.1]),
    showlegend=False, bargap=0.35,
)
save(fig, "09_silhouette_cluster", h=460)

# ═══════════════════════════════════════════════════════════════════════
# 10. CLUSTER DISTRIBUTION (Donut)
# ═══════════════════════════════════════════════════════════════════════
print("Chart 10: Cluster Distribution...")
cls_cnt = pd.Series(labels).value_counts().sort_index()
display_lbls = [f"Cụm {int(c)+1} – {CLUSTER_NAMES.get(int(c),'')}" for c in cls_cnt.index]
fig = go.Figure(go.Pie(
    labels=display_lbls, values=cls_cnt.values, hole=0.5,
    marker=dict(colors=[COLORS[i%len(COLORS)] for i in range(len(cls_cnt))],
                line=dict(color="#07111f", width=2)),
    textinfo="percent+label",
    textfont=dict(size=12, color="#fff"),
    hovertemplate="%{label}: %{value:,} tiêu đề (%{percent})<extra></extra>",
))
fig.update_layout(
    title=dict(text="Phân bố nội dung theo cụm (K=5)", font=dict(size=18,color=FONT_CLR), x=0.5),
    annotations=[dict(text=f"<b>{len(df):,}</b><br>Titles", x=0.5, y=0.5,
                      showarrow=False, font=dict(size=17,color=FONT_CLR))],
    height=500, paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
    font=dict(color=FONT_CLR, family="Arial"),
    margin=dict(l=30,r=30,t=60,b=30), showlegend=True,
    legend=dict(bgcolor="rgba(8,22,37,0.75)", bordercolor="rgba(255,255,255,0.08)",
                borderwidth=1, font=dict(color=FONT_CLR, size=12)),
)
save(fig, "10_cluster_donut", h=500)

# ═══════════════════════════════════════════════════════════════════════
# 11. GENRE HEATMAP BY CLUSTER
# ═══════════════════════════════════════════════════════════════════════
print("Chart 11: Genre Heatmap by Cluster...")
exploded = df[["cluster","genre_list"]].explode("genre_list")
counts_df = exploded.groupby(["genre_list","cluster"]).size().reset_index(name="cnt")
top12 = df["genre_list"].explode().value_counts().head(12).index.tolist()
cids  = sorted(pd.Series(labels).unique())

matrix = []
for g in top12:
    row = []
    for c in cids:
        v = counts_df.loc[(counts_df["genre_list"]==g)&(counts_df["cluster"]==c),"cnt"].sum()
        row.append(int(v))
    tot = max(sum(row),1)
    matrix.append([round(x/tot,3) for x in row])

fig = go.Figure(go.Heatmap(
    z=matrix,
    x=[f"C{c+1}" for c in cids],
    y=top12,
    colorscale=[[0,"#0d1b2a"],[0.25,"#7e174c"],[0.55,"#e82135"],[0.78,"#ff8f1f"],[1,"#ffd166"]],
    colorbar=dict(title=dict(text="Tỷ lệ", font=dict(color=FONT_CLR)), tickfont=dict(color=FONT_CLR)),
    hovertemplate="%{y}<br>%{x}: %{z:.1%}<extra></extra>",
    xgap=3, ygap=3,
))
fig.update_xaxes(side="top", color=FONT_CLR, tickfont=dict(size=13,color=FONT_CLR))
fig.update_yaxes(color=FONT_CLR, tickfont=dict(size=11))
fig.update_layout(
    title=dict(text="Sở thích thể loại theo từng cụm (Tỷ lệ tương đối)",
               font=dict(size=18,color=FONT_CLR), x=0.5),
    height=560, paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
    font=dict(color=FONT_CLR, family="Arial"),
    margin=dict(l=210,r=80,t=80,b=30), showlegend=False,
)
save(fig, "11_genre_heatmap", h=560)

# ═══════════════════════════════════════════════════════════════════════
# 12. COUNTRY DISTRIBUTION TOP10
# ═══════════════════════════════════════════════════════════════════════
print("Chart 12: Country Distribution...")
country_cnt = df["country_list"].explode().value_counts().head(10).sort_values()
fig = go.Figure(go.Bar(
    x=country_cnt.values, y=country_cnt.index,
    orientation="h",
    marker=dict(color=country_cnt.values, colorscale="ice_r",
                line=dict(color="rgba(255,255,255,0.08)",width=0.5)),
    text=[f"{v:,}" for v in country_cnt.values],
    textposition="outside", cliponaxis=False,
    hovertemplate="%{y}: %{x:,}<extra></extra>",
))
fig.update_layout(
    title=dict(text="Top 10 quốc gia sản xuất nội dung", font=dict(size=18,color=FONT_CLR), x=0.5),
    height=460, paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
    font=dict(color=FONT_CLR, family="Arial"),
    margin=dict(l=180,r=80,t=60,b=40),
    xaxis=dict(gridcolor=GRID, color=FONT_CLR, title="Số lượng tiêu đề"),
    yaxis=dict(gridcolor=GRID, color=FONT_CLR, tickfont=dict(size=12)),
    showlegend=False,
)
save(fig, "12_country_dist", h=460)

# ═══════════════════════════════════════════════════════════════════════
# 13. RELEASE YEAR TREND
# ═══════════════════════════════════════════════════════════════════════
print("Chart 13: Release Year Trend...")
year_df = df[df["release_year"]>=1990].groupby(["release_year","type"]).size().reset_index(name="cnt")
fig = go.Figure()
for t, col, fill_col in [("Movie","#e82135","rgba(232,33,53,0.12)"),("TV Show","#1976d2","rgba(25,118,210,0.12)")]:
    sub = year_df[year_df["type"]==t]
    fig.add_trace(go.Scatter(
        x=sub["release_year"], y=sub["cnt"],
        mode="lines+markers", name=t,
        line=dict(color=col, width=2.5),
        marker=dict(color=col, size=5),
        fill="tozeroy", fillcolor=fill_col,
        hovertemplate=f"{t} năm %{{x}}: %{{y:,}}<extra></extra>",
    ))
fig.update_layout(
    title=dict(text="Xu hướng sản xuất nội dung theo năm (từ 1990)", font=dict(size=18,color=FONT_CLR), x=0.5),
    height=460, paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
    font=dict(color=FONT_CLR, family="Arial"),
    margin=dict(l=60,r=30,t=70,b=55),
    xaxis=dict(gridcolor=GRID, color=FONT_CLR, title="Năm phát hành"),
    yaxis=dict(gridcolor=GRID, color=FONT_CLR, title="Số lượng"),
    showlegend=True,
    legend=dict(bgcolor="rgba(8,22,37,0.75)", bordercolor="rgba(255,255,255,0.08)",
                borderwidth=1, font=dict(color=FONT_CLR,size=13)),
)
save(fig, "13_year_trend", h=460)

print(f"\nDone! {len(list(OUT.glob('*.png')))} charts exported to {OUT}")
print(f"Silhouette Score (K=5): {sil_mean.mean():.4f}")
print(f"Cluster sizes: {dict(zip([f'C{i+1}' for i in range(5)], cls_cnt.values.tolist()))}")
