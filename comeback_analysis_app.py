import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Comeback Momentum Analysis",
    page_icon="🎵",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title(
    "🎵 Comeback Momentum, Chart Re-Entry & Fandom Intensity Analysis"
)

st.markdown(
    """
    ### South Korea Top 50 Playlist Analytics

    This dashboard analyzes chart re-entries, comeback momentum,
    retention, rank recovery, content attributes and a constructed
    Fandom Intensity Proxy.
    """
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("Atlantic_South_Korea.csv")

    df["date"] = pd.to_datetime(df["date"])

    return df


try:

    df = load_data()

except Exception as e:

    st.error("❌ Dataset could not be loaded.")

    st.write("Error:", e)

    st.info(
        "Make sure Atlantic_South_Korea.csv is in the "
        "same GitHub repository as this app."
    )

    st.stop()


# ============================================================
# PREPARE DATA
# ============================================================

df = df.copy()

# Song ID
df["song_id"] = (
    df["song"].astype(str).str.strip()
    + " — "
    + df["artist"].astype(str).str.strip()
)

# Sort
df = df.sort_values(
    ["song_id", "date"]
).reset_index(drop=True)

# Previous date
df["prev_date"] = (
    df.groupby("song_id")["date"]
    .shift(1)
)

# Previous position
df["prev_position"] = (
    df.groupby("song_id")["position"]
    .shift(1)
)

# Previous popularity
df["prev_popularity"] = (
    df.groupby("song_id")["popularity"]
    .shift(1)
)

# Gap
df["gap_days"] = (
    df["date"] -
    df["prev_date"]
).dt.days

# Entry type
df["entry_type"] = np.where(
    df["prev_date"].isna(),
    "First Entry",
    np.where(
        df["gap_days"] > 1,
        "Re-entry",
        "Continuing"
    )
)

# Re-entry flag
df["is_reentry"] = (
    df["entry_type"] == "Re-entry"
).astype(int)

# ============================================================
# MOMENTUM
# ============================================================

df["rank_jump"] = (
    df["prev_position"] -
    df["position"]
)

df["popularity_change"] = (
    df["popularity"] -
    df["prev_popularity"]
)

rank_component = np.clip(
    (
        df["rank_jump"].fillna(0) + 50
    ) / 100 * 100,
    0,
    100
)

pop_component = np.clip(
    (
        df["popularity_change"].fillna(0) + 20
    ) / 40 * 100,
    0,
    100
)

gap_component = np.clip(
    df["gap_days"].fillna(0) / 30 * 100,
    0,
    100
)

df["momentum_spike_score"] = (
    0.50 * rank_component
    + 0.35 * pop_component
    + 0.15 * gap_component
)

# ============================================================
# RE-ENTRY METRICS
# ============================================================

peak_rank_values = {}
retention_values = {}
recovery_values = {}

for song_id in df["song_id"].dropna().unique():

    song_data = (
        df[df["song_id"] == song_id]
        .sort_values("date")
    )

    for current_index, row in song_data.iterrows():

        if row["entry_type"] not in [
            "First Entry",
            "Re-entry"
        ]:
            continue

        start_date = row["date"]

        # 7-day peak rank
        seven_day_data = song_data[
            (song_data["date"] >= start_date)
            &
            (
                song_data["date"]
                <= start_date + pd.Timedelta(days=7)
            )
        ]

        if len(seven_day_data) > 0:
            peak_rank = int(
                seven_day_data["position"].min()
            )
        else:
            peak_rank = int(row["position"])

        # Retention
        later_data = song_data[
            song_data["date"] > start_date
        ]

        if len(later_data) > 0:

            last_date = later_data["date"].max()

            retention_days = int(
                (last_date - start_date).days
            )

        else:

            retention_days = 0

        # Recovery speed
        if (
            row["entry_type"] == "Re-entry"
            and pd.notna(row["gap_days"])
            and row["gap_days"] > 0
        ):

            rank_jump_value = max(
                float(row["rank_jump"]),
                0
            )

            recovery_speed = (
                rank_jump_value /
                float(row["gap_days"])
            )

        else:

            recovery_speed = 0.0

        peak_rank_values[current_index] = peak_rank
        retention_values[current_index] = retention_days
        recovery_values[current_index] = recovery_speed


df["peak_rank_7d"] = (
    df.index.map(peak_rank_values)
)

df["retention_days"] = (
    df.index.map(retention_values)
)

df["rank_recovery_speed"] = (
    df.index.map(recovery_values)
)

df["peak_rank_7d"] = (
    df["peak_rank_7d"]
    .fillna(df["position"])
)

df["retention_days"] = (
    df["retention_days"]
    .fillna(0)
)

df["rank_recovery_speed"] = (
    df["rank_recovery_speed"]
    .fillna(0)
)

# ============================================================
# RE-ENTRY FREQUENCY
# ============================================================

df["reentry_count"] = (
    df.groupby("song_id")["is_reentry"]
    .transform("sum")
)

# ============================================================
# FANDOM INTENSITY PROXY
# ============================================================

reentry_component = np.clip(
    df["reentry_count"] * 20,
    0,
    100
)

recovery_component = np.clip(
    df["rank_recovery_speed"] * 10,
    0,
    100
)

retention_component = np.clip(
    df["retention_days"] / 30 * 100,
    0,
    100
)

df["fandom_intensity_proxy"] = (
    0.40 * reentry_component
    + 0.30 * np.clip(
        df["momentum_spike_score"],
        0,
        100
    )
    + 0.15 * recovery_component
    + 0.15 * retention_component
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎛️ Dashboard Filters")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:

    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

else:

    start_date = pd.Timestamp(min_date)
    end_date = pd.Timestamp(max_date)


artists = sorted(
    df["artist"].dropna().astype(str).unique()
)

selected_artists = st.sidebar.multiselect(
    "Artist",
    artists
)


songs = sorted(
    df["song"].dropna().astype(str).unique()
)

selected_songs = st.sidebar.multiselect(
    "Song",
    songs
)


album_types = sorted(
    df["album_type"].dropna().astype(str).unique()
) if "album_type" in df.columns else []

selected_album_types = st.sidebar.multiselect(
    "Album Type",
    album_types
)

min_reentries = st.sidebar.slider(
    "Minimum Re-entry Count",
    0,
    10,
    0
)

# ============================================================
# APPLY FILTERS
# ============================================================

filtered = df[
    (df["date"] >= start_date)
    &
    (df["date"] <= end_date)
].copy()

if selected_artists:

    filtered = filtered[
        filtered["artist"].isin(selected_artists)
    ]

if selected_songs:

    filtered = filtered[
        filtered["song"].isin(selected_songs)
    ]

if selected_album_types:

    filtered = filtered[
        filtered["album_type"].isin(selected_album_types)
    ]

filtered = filtered[
    filtered["reentry_count"] >= min_reentries
]

if filtered.empty:

    st.warning(
        "⚠️ No data matches the selected filters."
    )

    st.stop()

# ============================================================
# COMEBACK DATA
# ============================================================

comeback = filtered[
    filtered["entry_type"] == "Re-entry"
].copy()

# ============================================================
# KPI VALUES
# ============================================================

total_songs = filtered["song_id"].nunique()

total_records = len(filtered)

total_reentries = len(comeback)

avg_momentum = (
    comeback["momentum_spike_score"].mean()
    if len(comeback) > 0
    else 0
)

avg_retention = (
    comeback["retention_days"].mean()
    if len(comeback) > 0
    else 0
)

# ============================================================
# KPI CARDS
# ============================================================

st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Songs",
    total_songs
)

col2.metric(
    "Records",
    f"{total_records:,}"
)

col3.metric(
    "Re-entry Events",
    total_reentries
)

col4.metric(
    "Avg Momentum",
    f"{avg_momentum:.2f}"
)

col5.metric(
    "Avg Retention",
    f"{avg_retention:.1f} days"
)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🏠 Overview",
        "🔄 Re-entry Explorer",
        "🚀 Momentum",
        "💿 Content Attributes",
        "🔥 Fandom Proxy",
        "🔍 Data Quality"
    ]
)

# ============================================================
# OVERVIEW
# ============================================================

with tab1:

    st.header("Overview")

    daily_entries = (
        filtered.groupby("date")
        .size()
        .reset_index(name="entries")
    )

    fig = px.line(
        daily_entries,
        x="date",
        y="entries",
        title="Daily Playlist Entries"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Top Artists by Chart Entries"
    )

    artist_counts = (
        filtered.groupby("artist")
        .size()
        .reset_index(name="entries")
        .sort_values(
            "entries",
            ascending=False
        )
        .head(15)
    )

    fig = px.bar(
        artist_counts,
        x="entries",
        y="artist",
        orientation="h",
        title="Top 15 Artists"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# RE-ENTRY EXPLORER
# ============================================================

with tab2:

    st.header("🔄 Chart Re-entry Explorer")

    if len(comeback) == 0:

        st.warning(
            "No re-entry events found."
        )

    else:

        fig = px.scatter(
            comeback,
            x="date",
            y="position",
            color="artist",
            hover_data=[
                "song",
                "artist",
                "gap_days",
                "position"
            ],
            title="Re-entry Timeline"
        )

        fig.update_yaxes(
            autorange="reversed"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Songs with Most Re-entries"
        )

        leaderboard = (
            comeback.groupby(
                ["song", "artist"]
            )
            .size()
            .reset_index(
                name="reentry_events"
            )
            .sort_values(
                "reentry_events",
                ascending=False
            )
            .head(20)
        )

        st.dataframe(
            leaderboard,
            use_container_width=True
        )

# ============================================================
# MOMENTUM
# ============================================================

with tab3:

    st.header("🚀 Comeback Momentum")

    if len(comeback) == 0:

        st.warning(
            "No comeback events available."
        )

    else:

        fig = px.histogram(
            comeback,
            x="momentum_spike_score",
            nbins=20,
            title="Momentum Spike Score Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig = px.scatter(
            comeback,
            x="momentum_spike_score",
            y="retention_days",
            color="artist",
            hover_data=[
                "song",
                "artist",
                "position"
            ],
            title="Momentum vs Retention"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "🔥 Strongest Comeback Events"
        )

        momentum_table = (
            comeback[
                [
                    "date",
                    "song",
                    "artist",
                    "position",
                    "gap_days",
                    "momentum_spike_score",
                    "retention_days"
                ]
            ]
            .sort_values(
                "momentum_spike_score",
                ascending=False
            )
            .head(20)
        )

        st.dataframe(
            momentum_table,
            use_container_width=True
        )

# ============================================================
# CONTENT ATTRIBUTES
# ============================================================

with tab4:

    st.header(
        "💿 Content Attributes vs Momentum"
    )

    if "album_type" in filtered.columns:

        album_analysis = (
            filtered.groupby("album_type")
            .agg(
                average_popularity=(
                    "popularity",
                    "mean"
                ),
                average_position=(
                    "position",
                    "mean"
                ),
                entries=(
                    "song_id",
                    "count"
                )
            )
            .reset_index()
        )

        fig = px.bar(
            album_analysis,
            x="album_type",
            y="average_popularity",
            title="Average Popularity by Album Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if "is_explicit" in filtered.columns:

        explicit_analysis = (
            filtered.groupby("is_explicit")
            .size()
            .reset_index(name="entries")
        )

        explicit_analysis["content_type"] = (
            explicit_analysis["is_explicit"]
            .map({
                True: "Explicit",
                False: "Clean",
                1: "Explicit",
                0: "Clean"
            })
        )

        fig = px.pie(
            explicit_analysis,
            names="content_type",
            values="entries",
            title="Explicit vs Clean Content"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ============================================================
# FANDOM PROXY
# ============================================================

with tab5:

    st.header(
        "🔥 Fandom Intensity Proxy"
    )

    st.info(
        "This is a constructed analytical proxy based on "
        "observable chart behavior. It does not directly "
        "measure real-world fandom activity."
    )

    fandom = (
        filtered.groupby(
            ["song", "artist"]
        )
        .agg(
            fandom_intensity=(
                "fandom_intensity_proxy",
                "max"
            ),
            reentries=(
                "is_reentry",
                "sum"
            ),
            avg_momentum=(
                "momentum_spike_score",
                "mean"
            )
        )
        .reset_index()
        .sort_values(
            "fandom_intensity",
            ascending=False
        )
        .head(20)
    )

    fig = px.bar(
        fandom,
        x="fandom_intensity",
        y="song",
        color="artist",
        orientation="h",
        title="Fandom Intensity Proxy"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        fandom,
        use_container_width=True
    )

# ============================================================
# DATA QUALITY
# ============================================================

with tab6:

    st.header(
        "🔍 Data Quality"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        len(filtered)
    )

    col2.metric(
        "Columns",
        len(filtered.columns)
    )

    col3.metric(
        "Unique Songs",
        filtered["song_id"].nunique()
    )

    st.subheader(
        "Missing Values"
    )

    missing = (
        filtered.isnull()
        .sum()
        .reset_index()
    )

    missing.columns = [
        "column",
        "missing_values"
    ]

    missing = missing[
        missing["missing_values"] > 0
    ]

    if len(missing) > 0:

        st.dataframe(
            missing,
            use_container_width=True
        )

    else:

        st.success(
            "✅ No missing values detected."
        )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        filtered.head(100),
        use_container_width=True
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Comeback Momentum, Chart Re-Entry, and Fandom Intensity "
    "Analysis of South Korea Top 50 Playlist"
)

st.caption(
    "Python • Pandas • NumPy • Plotly • Streamlit"
)
