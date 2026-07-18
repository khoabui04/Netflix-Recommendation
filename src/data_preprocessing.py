import pandas as pd


def load_netflix_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def parse_duration(duration: str) -> dict:
    if pd.isna(duration):
        return {"duration_min": 0, "season_count": 0}
    value = duration.strip()
    if "min" in value:
        try:
            minutes = int(value.replace("min", "").strip())
        except ValueError:
            minutes = 0
        return {"duration_min": minutes, "season_count": 0}
    if "Season" in value or "Seasons" in value:
        try:
            season_count = int(value.split(" ")[0])
        except ValueError:
            season_count = 0
        return {"duration_min": 0, "season_count": season_count}
    return {"duration_min": 0, "season_count": 0}


def preprocess_netflix_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["show_id"])
    df["type"] = df["type"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["listed_in"] = df["listed_in"].fillna("Unknown")
    df["director"] = df["director"].fillna("Unknown")
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)

    duration_parsed = df["duration"].apply(parse_duration).apply(pd.Series)
    df = pd.concat([df, duration_parsed], axis=1)
    df["duration_min"] = df["duration_min"].fillna(0).astype(int)
    df["season_count"] = df["season_count"].fillna(0).astype(int)

    rating_order = {
        "G": 1,
        "PG": 2,
        "PG-13": 3,
        "TV-Y": 1,
        "TV-Y7": 1,
        "TV-G": 1,
        "TV-PG": 2,
        "TV-14": 3,
        "TV-MA": 4,
        "R": 4,
        "NR": 0,
        "UR": 0,
        "Unknown": 0,
    }
    df["rating"] = df["rating"].fillna("Unknown")
    df["rating_level"] = df["rating"].map(rating_order).fillna(0).astype(int)

    df["genre_list"] = df["listed_in"].astype(str).apply(lambda text: [g.strip() for g in text.split(",") if g.strip()])
    df["country_list"] = df["country"].astype(str).apply(lambda text: [c.strip() for c in text.split(",") if c.strip()])
    df["title"] = df["title"].astype(str)
    return df
