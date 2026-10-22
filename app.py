from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")

# ============================================================
# RESTAURANT INTELLIGENCE | HARSHITA MISHRA
# ============================================================

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "Dataset.csv"
RATING_MODEL_FILE = BASE / "restaurant_rating_model.pkl"
CUISINE_MODEL_FILE = BASE / "cuisine_classification_model.pkl"

st.set_page_config(
    page_title="Restaurant Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- THEME -----------------------------

if "ri_theme" not in st.session_state:
    st.session_state.ri_theme = "Dark"

DARK = st.session_state.ri_theme == "Dark"

if DARK:
    C = {
        "bg": "#0B1020",
        "panel": "#131B30",
        "panel2": "#19233B",
        "text": "#F3F5FF",
        "muted": "#A6B1CC",
        "border": "#2A3550",
        "purple": "#A78BFA",
        "purple2": "#7955E8",
        "grid": "#29334A",
        "chart": "#131B30",
        "input": "#19233B",
    }
else:
    C = {
        "bg": "#F5F6FC",
        "panel": "#FFFFFF",
        "panel2": "#F0EDFF",
        "text": "#19233B",
        "muted": "#65728B",
        "border": "#E1E5F0",
        "purple": "#6941D9",
        "purple2": "#8059ED",
        "grid": "#E8EAF3",
        "chart": "#FFFFFF",
        "input": "#FFFFFF",
    }

PURPLE = C["purple"]
PALETTE = [
    "#8B5CF6", "#2DD4BF", "#F59E0B", "#60A5FA",
    "#F472B6", "#A78BFA", "#34D399", "#FB7185",
]

# ------------------------- DESIGN ----------------------------

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {{
        color-scheme: {"dark" if DARK else "light"};
    }}

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
    }}

    .stApp {{
        background: {C["bg"]};
        color: {C["text"]};
    }}

    [data-testid="stHeader"] {{
        background: {C["bg"]};
    }}

    [data-testid="stMainBlockContainer"] {{
        max-width: 1550px;
        padding: 2rem 2.5rem 3rem;
    }}

    h1, h2, h3, h4 {{
        font-family: 'Manrope', sans-serif !important;
        color: {C["text"]} !important;
        letter-spacing: -0.5px;
    }}

    p, li, label, .stCaption {{
        color: {C["muted"]};
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(160deg, #10172A, #171F38 65%, #242449);
        border-right: 1px solid #303650;
    }}

    [data-testid="stSidebar"] > div:first-child {{
        padding: 1.4rem 1rem;
    }}

    [data-testid="stSidebar"] * {{
        color: #F2F2FF;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: #B6C0D8;
    }}

    [data-testid="stSidebar"] hr {{
        border-color: #35405A;
        margin: 1.2rem 0;
    }}

    .brand {{
        font-family: 'Manrope', sans-serif;
        font-size: 1.5rem;
        line-height: 1.35;
        font-weight: 800;
        color: white;
    }}

    .brand span {{
        color: #B5A0FF;
    }}

    .brand-sub {{
        color: #A6B1CC;
        font-size: .68rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 8px;
    }}

    .side-label {{
        color: #9AA7C6;
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: 1.8px;
        margin: 1rem 0 .6rem;
    }}

    [data-testid="stSidebar"] .stRadio label {{
        padding: 7px 9px;
        border-radius: 10px;
        transition: background .2s ease;
    }}

    [data-testid="stSidebar"] .stRadio label:hover {{
        background: #2B3453;
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(115deg, #151D35 0%, #352969 58%, #7652E8 100%);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 22px;
        padding: 2.2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 14px 40px rgba(40,25,95,.18);
    }}

    .hero-eyebrow {{
        color: #D8CCFF;
        font-size: .7rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: .8rem;
    }}

    .hero h1 {{
        color: #FFFFFF !important;
        font-size: clamp(1.8rem, 3vw, 2.8rem);
        line-height: 1.2;
        margin: 0 0 .7rem;
    }}

    .hero p {{
        color: #E0DDF4 !important;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 760px;
        margin-bottom: 0;
    }}

    .section-kicker {{
        color: {PURPLE};
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: .3rem;
    }}

    .section-note {{
        color: {C["muted"]};
        font-size: .9rem;
        margin-top: -.3rem;
        margin-bottom: 1.1rem;
    }}

    [data-testid="stMetric"] {{
        background: {C["panel"]};
        border: 1px solid {C["border"]};
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        min-height: 115px;
        box-shadow: 0 5px 20px rgba(0,0,0,.04);
    }}

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {{
        color: {C["muted"]} !important;
        font-size: .82rem !important;
        font-weight: 600;
    }}

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {{
        color: {C["text"]} !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 800 !important;
    }}

    [data-testid="stPlotlyChart"] {{
        background: {C["panel"]};
        border: 1px solid {C["border"]};
        border-radius: 16px;
        padding: 8px;
    }}

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {{
        border: 1px solid {C["border"]};
        border-radius: 12px;
        overflow: hidden;
    }}

    [data-testid="stExpander"] {{
        background: {C["panel"]};
        border: 1px solid {C["border"]};
        border-radius: 12px;
    }}

    .stButton > button,
    .stFormSubmitButton > button,
    [data-testid="stDownloadButton"] > button {{
        background: linear-gradient(110deg, #6941D9, #906BFA);
        color: white !important;
        border: none;
        border-radius: 10px;
        min-height: 43px;
        font-weight: 700;
        transition: all .2s ease;
    }}

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    [data-testid="stDownloadButton"] > button:hover {{
        background: #5733BD;
        border: none;
        transform: translateY(-1px);
    }}

    .stSelectbox label,
    .stNumberInput label,
    .stSlider label,
    .stMultiSelect label,
    .stSelectSlider label {{
        color: {C["text"]} !important;
        font-weight: 600;
    }}

    [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input {{
        background: {C["input"]};
        color: {C["text"]} !important;
        border-color: {C["border"]};
        border-radius: 9px;
    }}

    .info-card {{
        background: {C["panel"]};
        border: 1px solid {C["border"]};
        border-radius: 16px;
        padding: 1.3rem;
        min-height: 150px;
        height: 100%;
        box-shadow: 0 5px 18px rgba(0,0,0,.025);
    }}

    .info-icon {{
        font-size: 1.5rem;
        margin-bottom: .7rem;
    }}

    .info-card h3 {{
        font-size: 1.02rem;
        margin: 0 0 .5rem;
    }}

    .info-card p {{
        color: {C["muted"]};
        font-size: .87rem;
        line-height: 1.65;
        margin-bottom: 0;
    }}

    .result-card {{
        background: {C["panel"]};
        border: 1px solid {C["border"]};
        border-left: 4px solid {PURPLE};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: .5rem 0;
    }}

    .pill {{
        display: inline-block;
        background: {C["panel2"]};
        border: 1px solid {C["border"]};
        color: {C["text"]};
        padding: 6px 11px;
        border-radius: 999px;
        font-size: .75rem;
        font-weight: 700;
        margin: 3px;
    }}

    .footer {{
        border-top: 1px solid {C["border"]};
        margin-top: 3rem;
        padding-top: 1.2rem;
        color: {C["muted"]};
        font-size: .76rem;
    }}

    @media(max-width: 800px) {{
        [data-testid="stMainBlockContainer"] {{
            padding: 1rem;
        }}
        .hero {{
            padding: 1.4rem;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- HELPERS ----------------------------

def hero(eyebrow, title, subtitle):
    st.markdown(
        f'<div class="hero"><div class="hero-eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section(title, note=None):
    st.markdown(
        f'<div class="section-kicker">EXPLORE</div><h2>{title}</h2>',
        unsafe_allow_html=True,
    )
    if note:
        st.markdown(f'<div class="section-note">{note}</div>',
                    unsafe_allow_html=True)


def info_card(icon, title, description):
    st.markdown(
        f'<div class="info-card"><div class="info-icon">{icon}</div>'
        f'<h3>{title}</h3><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def chart_layout(fig, height=350):
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=C["chart"],
        font=dict(family="DM Sans, sans-serif",
                  color=C["text"], size=12),
        title_font=dict(family="Manrope, sans-serif",
                        size=16, color=C["text"]),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor=C["panel2"],
                        font_color=C["text"]),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=C["border"],
        tickfont=dict(color=C["muted"]),
        title_font=dict(color=C["muted"]),
    )
    fig.update_yaxes(
        gridcolor=C["grid"],
        zeroline=False,
        tickfont=dict(color=C["muted"]),
        title_font=dict(color=C["muted"]),
    )
    return fig


def safe_model(path):
    if not path.exists():
        return None, f"Model file not found: {path.name}"
    try:
        return joblib.load(path), None
    except Exception as exc:
        return None, f"Could not load {path.name}: {exc}"


@st.cache_data
def load_data(path, modified_time):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    return data


# ------------------------- LOAD DATA ---------------------------

if not DATA_FILE.exists():
    st.error("Dataset.csv must be in the same folder as app.py.")
    st.stop()

try:
    df = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not read Dataset.csv: {exc}")
    st.stop()

required = {
    "Restaurant Name", "City", "Cuisines", "Aggregate rating",
    "Votes", "Price range", "Average Cost for two",
}
missing = required - set(df.columns)
if missing:
    st.error("Dataset is missing columns: " + ", ".join(sorted(missing)))
    st.stop()

rating_model, rating_error = safe_model(RATING_MODEL_FILE)
cuisine_model, cuisine_error = safe_model(CUISINE_MODEL_FILE)

rated_df = df[df["Aggregate rating"].fillna(0) > 0].copy()
cuisine_df = df.dropna(subset=["Cuisines"]).copy()
cuisine_df["Primary Cuisine"] = (
    cuisine_df["Cuisines"].astype(str).str.split(",").str[0].str.strip()
)

cities = sorted(df["City"].dropna().astype(str).unique())
cuisines = sorted(df["Cuisines"].dropna().astype(str).unique())

# ------------------------- SIDEBAR ----------------------------

with st.sidebar:
    st.markdown(
        '<div class="brand">🍽️ Restaurant<br><span>Intelligence</span></div>'
        '<div class="brand-sub">DATA MEETS DINING</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown('<div class="side-label">APPEARANCE</div>',
                unsafe_allow_html=True)

    new_theme = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=0 if DARK else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="ri_theme_choice",
    )

    if new_theme != st.session_state.ri_theme:
        st.session_state.ri_theme = new_theme
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="side-label">WORKSPACE</div>',
                unsafe_allow_html=True)

    pages = {
        "Dashboard": "◈",
        "Rating Prediction": "✦",
        "Restaurant Recommendation": "⌕",
        "Cuisine Classification": "◇",
        "Location Analysis": "⌖",
        "Data Insights": "▥",
        "Dataset Info": "▤",
        "About Project": "ⓘ",
    }

    page = st.radio(
        "Navigation",
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]}   {x}",
        label_visibility="collapsed",
        key="ri_page",
    )

    st.markdown("---")
    st.markdown('<div class="side-label">PROJECT OWNER</div>',
                unsafe_allow_html=True)
    st.markdown("**Harshita Mishra**")
    st.caption("Machine Learning Developer")
    st.markdown("---")
    st.caption("Python · Pandas · Scikit-learn")
    st.caption("Streamlit · Plotly · Joblib")

# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":
    hero(
        "YOUR RESTAURANT ANALYTICS WORKSPACE",
        "Restaurant Intelligence",
        "A smarter way to explore restaurant ratings, discover dining options, "
        "and uncover patterns in cuisine, location, and pricing.",
    )

    total = len(df)
    rated_count = len(rated_df)
    avg_rating = rated_df["Aggregate rating"].mean() if rated_count else 0

    a, b, c, d = st.columns(4)
    a.metric("Restaurants", f"{total:,}")
    b.metric("Rated restaurants", f"{rated_count:,}")
    c.metric("Cities covered", f"{df['City'].nunique():,}")
    d.metric("Average rating", f"{avg_rating:.2f} / 5")

    st.markdown("")
    section("The big picture", "Explore the patterns hidden in the dataset.")

    left, right = st.columns(2)

    with left:
        counts = df["City"].value_counts().head(10).sort_values()
        fig = px.bar(
            x=counts.values,
            y=counts.index,
            orientation="h",
            title="Top 10 cities by restaurant count",
            labels={"x": "Restaurants", "y": ""},
            color=counts.values,
            color_continuous_scale=["#C4B5FD", "#7048E8"],
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(chart_layout(fig, 390), use_container_width=True)

    with right:
        primary = cuisine_df["Primary Cuisine"].value_counts().head(8)
        fig = px.pie(
            names=primary.index,
            values=primary.values,
            hole=.62,
            title="Most common primary cuisines",
            color_discrete_sequence=PALETTE,
        )
        fig.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(chart_layout(fig, 390), use_container_width=True)

    left, right = st.columns(2)

    with left:
        if not rated_df.empty:
            fig = px.histogram(
                rated_df,
                x="Aggregate rating",
                nbins=20,
                title="Restaurant rating distribution",
                labels={"Aggregate rating": "Rating"},
                color_discrete_sequence=[PURPLE],
            )
            st.plotly_chart(chart_layout(fig, 330),
                            use_container_width=True)

    with right:
        prices = df["Price range"].value_counts().sort_index()
        fig = px.bar(
            x=prices.index.astype(str),
            y=prices.values,
            title="Restaurants by price range",
            labels={"x": "Price range", "y": "Restaurants"},
            color=prices.index.astype(str),
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_layout(fig, 330),
                        use_container_width=True)

    section("Explore your workspace", "Choose a module from the sidebar.")
    cards = [
        ("✦", "Rating Prediction", "Estimate restaurant ratings with a trained regression model."),
        ("🍜", "Restaurant Recommendation", "Discover restaurants matching your dining preferences."),
        ("🧠", "Cuisine Classification", "Predict a likely primary cuisine category."),
        ("📍", "Location Analysis", "Explore restaurant locations and city-level patterns."),
    ]

    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, cards):
        with col:
            info_card(icon, title, desc)

# ============================================================
# RATING PREDICTION
# ============================================================

elif page == "Rating Prediction":
    hero("MACHINE LEARNING · REGRESSION", "Rating Prediction",
         "Estimate a restaurant's aggregate rating using its characteristics.")

    if rating_model is None:
        st.error(rating_error)
        st.info("Place restaurant_rating_model.pkl next to app.py.")
    else:
        with st.form("rating_form"):
            section("Restaurant profile",
                    "Enter the details used by the rating prediction model.")

            c1, c2 = st.columns(2)

            with c1:
                city = st.selectbox("City", cities)
                cuisine = st.selectbox("Cuisine", cuisines)
                price_range = st.select_slider(
                    "Price range", options=[1, 2, 3, 4], value=2
                )

            with c2:
                votes = st.number_input(
                    "Expected / existing votes",
                    min_value=0, value=100, step=10,
                )
                booking = st.selectbox("Has table booking?", ["Yes", "No"])
                delivery = st.selectbox("Has online delivery?", ["Yes", "No"])

            submitted = st.form_submit_button(
                "✦  Predict restaurant rating",
                use_container_width=True,
            )

        if submitted:
            input_df = pd.DataFrame([{
                "Price range": price_range,
                "Votes": votes,
                "Has Table booking": booking,
                "Has Online delivery": delivery,
                "City": city,
                "Cuisines": cuisine,
            }])

            try:
                prediction = float(rating_model.predict(input_df)[0])
                prediction = max(0.0, min(5.0, prediction))

                st.markdown("---")
                section("Your prediction")
                a, b = st.columns([1, 2])

                with a:
                    st.metric("Estimated rating", f"{prediction:.2f} / 5")

                with b:
                    st.progress(prediction / 5)
                    st.caption("Predicted rating on a scale from 0 to 5.")

                st.info("This is a model estimate, not an official restaurant rating.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.caption("Check that the saved model matches the input features.")

# ============================================================
# RESTAURANT RECOMMENDATION
# ============================================================

elif page == "Restaurant Recommendation":
    hero("DISCOVER YOUR NEXT FAVOURITE", "Restaurant Recommendation",
         "Find restaurants based on cuisine, location, budget, and rating.")

    cuisine_options = sorted({
        part.strip()
        for value in cuisine_df["Cuisines"].astype(str)
        for part in value.split(",") if part.strip()
    })
    city_options = ["All Cities"] + cities

    with st.form("recommendation_form"):
        section("Your dining preferences")
        a, b = st.columns(2)

        with a:
            chosen_cuisine = st.selectbox("Cuisine preference", cuisine_options)
            chosen_city = st.selectbox("Location", city_options)

        with b:
            chosen_price = st.select_slider(
                "Maximum price range", options=[1, 2, 3, 4], value=2
            )
            minimum_rating = st.slider(
                "Minimum rating", 0.0, 5.0, 3.5, 0.5
            )

        count = st.select_slider(
            "Number of recommendations", options=[5, 10, 15, 20], value=10
        )

        find = st.form_submit_button(
            "⌕  Find restaurants", use_container_width=True
        )

    if find:
        results = df[
            df["Cuisines"].fillna("").str.contains(
                chosen_cuisine, case=False, regex=False
            )
        ].copy()

        if chosen_city != "All Cities":
            results = results[results["City"].astype(str) == chosen_city]

        results = results[
            (results["Price range"].fillna(99) <= chosen_price)
            & (results["Aggregate rating"].fillna(0) >= minimum_rating)
        ]

        sort_cols = [
            col for col in ["Aggregate rating", "Votes"]
            if col in results.columns
        ]
        results = results.sort_values(sort_cols, ascending=False)
        results = results.drop_duplicates(["Restaurant Name", "City"])

        st.markdown("---")
        st.metric("Matching restaurants", f"{len(results):,}")

        display_cols = [
            "Restaurant Name", "City", "Cuisines", "Price range",
            "Aggregate rating", "Votes", "Average Cost for two", "Currency",
        ]
        display_cols = [col for col in display_cols if col in results.columns]
        results = results[display_cols].head(count)

        if results.empty:
            st.warning("No matches found. Try a lower rating or a higher budget.")
        else:
            for _, row in results.iterrows():
                st.markdown(
                    f'<div class="result-card">'
                    f'<strong>{row.get("Restaurant Name", "Restaurant")}</strong>'
                    f'<br><span style="color:{C["muted"]}">'
                    f'{row.get("City", "Unknown city")} · '
                    f'{row.get("Cuisines", "Cuisine not listed")}</span>'
                    f'<br><br>⭐ {row.get("Aggregate rating", "N/A")} / 5'
                    f' &nbsp; · &nbsp; Price range: {row.get("Price range", "N/A")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with st.expander("View complete recommendation table"):
                st.dataframe(results, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇ Download recommendations as CSV",
                results.to_csv(index=False).encode("utf-8"),
                file_name="restaurant_recommendations.csv",
                mime="text/csv",
            )

    with st.expander("How recommendations work"):
        st.write(
            "This is a transparent, filter-based recommendation approach. "
            "It matches cuisine, optional city, maximum price range, and minimum rating, "
            "then ranks matches by rating and votes. It does not claim to learn personal tastes."
        )

# ============================================================
# CUISINE CLASSIFICATION
# ============================================================

elif page == "Cuisine Classification":
    hero("MACHINE LEARNING · CLASSIFICATION", "Cuisine Classification",
         "Predict a likely primary cuisine category from restaurant details.")

    if cuisine_model is None:
        st.error(cuisine_error)
        st.info("Place cuisine_classification_model.pkl next to app.py.")
    else:
        names = sorted(df["Restaurant Name"].dropna().astype(str).unique())
        countries = (
            sorted(df["Country Code"].dropna().unique())
            if "Country Code" in df.columns else [1]
        )

        with st.form("cuisine_form"):
            section("Restaurant profile",
                    "Enter restaurant details to predict its primary cuisine.")

            a, b = st.columns(2)

            with a:
                restaurant_name = st.selectbox("Restaurant name", names)
                city = st.selectbox("City", cities)
                price_range = st.select_slider(
                    "Price range", options=[1, 2, 3, 4], value=2
                )
                cost = st.number_input(
                    "Average cost for two",
                    min_value=0.0, value=700.0, step=50.0,
                )

            with b:
                country = st.selectbox("Country code", countries)
                booking = st.selectbox("Has table booking?", ["Yes", "No"])
                delivery = st.selectbox("Has online delivery?", ["Yes", "No"])
                delivering = st.selectbox("Is delivering now?", ["Yes", "No"])

            submitted = st.form_submit_button(
                "🧠  Predict cuisine", use_container_width=True
            )

        if submitted:
            input_df = pd.DataFrame([{
                "Restaurant Name": restaurant_name,
                "Price range": price_range,
                "Average Cost for two": cost,
                "Has Table booking": booking,
                "Has Online delivery": delivery,
                "Is delivering now": delivering,
                "City": city,
                "Country Code": country,
            }])

            try:
                predicted = cuisine_model.predict(input_df)[0]
                section("Classification result")
                st.success(f"Predicted primary cuisine: **{predicted}**")
                st.caption(
                    "This is a model estimate. The classifier can predict only "
                    "the cuisine classes represented in its training data."
                )
            except Exception as exc:
                st.error(f"Classification failed: {exc}")
                st.caption("Check that the saved model matches the input features.")

# ============================================================
# LOCATION ANALYSIS
# ============================================================

elif page == "Location Analysis":
    hero("GEOGRAPHIC EXPLORATION", "Location Analysis",
         "Explore restaurant concentrations, city-level ratings, and coordinates.")

    counts = df["City"].value_counts().head(15).sort_values()
    city_ratings = (
        rated_df.groupby("City")["Aggregate rating"]
        .agg(["mean", "count"])
        .query("count >= 5")
        .sort_values("mean", ascending=False)
        .head(15)
        .sort_values("mean")
    )

    a, b = st.columns(2)

    with a:
        fig = px.bar(
            x=counts.values, y=counts.index, orientation="h",
            title="Top 15 cities by restaurant count",
            labels={"x": "Restaurants", "y": ""},
            color=counts.values,
            color_continuous_scale=["#C4B5FD", "#7048E8"],
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(chart_layout(fig, 430), use_container_width=True)

    with b:
        if not city_ratings.empty:
            fig = px.bar(
                x=city_ratings["mean"], y=city_ratings.index,
                orientation="h",
                title="Average city rating · minimum 5 rated restaurants",
                labels={"x": "Average rating", "y": ""},
                color=city_ratings["mean"],
                color_continuous_scale=["#C4B5FD", "#7048E8"],
            )
            fig.update_layout(coloraxis_showscale=False, xaxis_range=[0, 5])
            st.plotly_chart(chart_layout(fig, 430), use_container_width=True)

    if {"Latitude", "Longitude"}.issubset(df.columns):
        geo = df.dropna(subset=["Latitude", "Longitude"]).copy()
        geo = geo[
            geo["Latitude"].between(-90, 90)
            & geo["Longitude"].between(-180, 180)
        ]

        if not geo.empty:
            section("Restaurant coordinate map",
                    "A sample of restaurants with available coordinates.")

            geo = geo.sample(min(1200, len(geo)), random_state=42)

            try:
                fig = px.scatter_map(
                    geo,
                    lat="Latitude",
                    lon="Longitude",
                    hover_name="Restaurant Name",
                    hover_data=["City", "Aggregate rating"],
                    color="Aggregate rating",
                    color_continuous_scale=["#C4B5FD", "#7955E8", "#2DD4BF"],
                    zoom=1,
                    height=520,
                )
                fig.update_layout(
                    map_style="open-street-map",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=10, b=0),
                    font=dict(color=C["text"]),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.warning(f"The coordinate map could not be rendered: {exc}")
                st.info(
                    "The city charts above are still available. "
                    "Update Plotly if you want to use the interactive map."
                )
        else:
            st.info("No usable latitude/longitude values were found.")

# ============================================================
# DATA INSIGHTS
# ============================================================

elif page == "Data Insights":
    hero("EXPLORE THE DATA", "Data Insights",
         "Investigate distributions, relationships, summary statistics, and missing values.")

    section("Numerical and categorical summary",
            "Statistical overview of the available dataset columns.")
    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True,
    )

    section("Missing values", "Columns with missing data.")
    missing_values = df.isna().sum().sort_values(ascending=False)
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        st.success("No missing values found.")
    else:
        fig = px.bar(
            x=missing_values.values,
            y=missing_values.index,
            orientation="h",
            title="Missing values by column",
            labels={"x": "Missing rows", "y": ""},
            color=missing_values.values,
            color_continuous_scale=["#C4B5FD", "#7048E8"],
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(
            chart_layout(fig, max(320, 32 * len(missing_values))),
            use_container_width=True,
        )

    section("Numeric relationships", "Explore correlations between numerical features.")
    numeric = df.select_dtypes(include=np.number)

    if numeric.shape[1] >= 2:
        corr = numeric.corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=["#EFDAFF", C["chart"], "#7955E8"],
            zmin=-1,
            zmax=1,
            title="Correlation matrix",
        )
        st.plotly_chart(chart_layout(fig, 600), use_container_width=True)

    section("Rating by price range")
    if not rated_df.empty:
        grouped = (
            rated_df.groupby("Price range")["Aggregate rating"]
            .mean().reset_index()
        )
        fig = px.bar(
            grouped,
            x="Price range",
            y="Aggregate rating",
            labels={"Aggregate rating": "Average rating"},
            color="Price range",
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(showlegend=False, yaxis_range=[0, 5])
        st.plotly_chart(chart_layout(fig, 330), use_container_width=True)

# ============================================================
# DATASET INFO
# ============================================================

elif page == "Dataset Info":
    hero("KNOW YOUR DATA", "Dataset Information",
         "Explore dataset dimensions, field types, completeness, and sample records.")

    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{df.shape[0]:,}")
    b.metric("Columns", f"{df.shape[1]:,}")
    c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    d.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    section("Column details", "Data types, missing cells, and unique values.")
    details = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[col].dtype) for col in df.columns],
        "Missing values": [int(df[col].isna().sum()) for col in df.columns],
        "Unique values": [int(df[col].nunique(dropna=True)) for col in df.columns],
    })
    st.dataframe(details, use_container_width=True, hide_index=True)

    section("Dataset preview", "Inspect a sample of the source records.")
    rows = st.slider("Rows to preview", 5, 50, 10, 5)
    st.dataframe(df.head(rows), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download dataset CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="restaurant_dataset.csv",
        mime="text/csv",
    )

# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "About Project":
    hero("THE STORY BEHIND THE PROJECT",
         "Turning restaurant data into insight",
         "A machine learning internship project developed by Harshita Mishra.")

    st.markdown(
        f"""
        <div style="background:{C['panel']};border:1px solid {C['border']};
                    border-radius:20px;padding:30px;margin-bottom:22px;">
            <div style="font-size:.7rem;letter-spacing:2px;text-transform:uppercase;
                        color:{PURPLE};font-weight:800;">FEATURED PROJECT</div>
            <h2 style="margin-top:12px;">🍽️ Restaurant Intelligence &amp;
            Recommendation System</h2>
            <p style="color:{C['muted']};font-size:1rem;line-height:1.9;">
                An interactive analytics application combining data exploration,
                supervised machine learning, restaurant recommendations, and
                geographic insights in one workspace.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("Project modules", "Explore the capabilities built into this application.")

    modules = [
        ("✦", "Rating Prediction",
         "Estimates aggregate ratings using a trained regression model."),
        ("🍜", "Restaurant Recommendation",
         "Filters restaurants by cuisine, location, budget, and minimum rating."),
        ("🧠", "Cuisine Classification",
         "Predicts a primary cuisine category using a trained classifier."),
        ("📍", "Location Analysis",
         "Explores city-level patterns and available restaurant coordinates."),
        ("📊", "Data Insights",
         "Displays distributions, summary statistics, correlations, and missing values."),
        ("🗂️", "Dataset Info",
         "Shows dataset dimensions, data types, uniqueness, and sample records."),
    ]

    for start in range(0, len(modules), 3):
        cols = st.columns(3)
        for col, item in zip(cols, modules[start:start + 3]):
            with col:
                info_card(*item)
        st.markdown("")

    section("Technology stack")
    st.markdown(
        '<span class="pill">Python</span>'
        '<span class="pill">Pandas</span>'
        '<span class="pill">NumPy</span>'
        '<span class="pill">Streamlit</span>'
        '<span class="pill">Plotly</span>'
        '<span class="pill">Scikit-learn</span>'
        '<span class="pill">Joblib</span>',
        unsafe_allow_html=True,
    )

    section("Model evaluation",
            "Previously measured evaluation results for the trained models.")

    a, b = st.columns(2)

    with a:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>✦ Rating Prediction</h3>
                <p>Random Forest regression evaluation</p>
                <p style="font-size:1rem;line-height:2;margin-top:12px;">
                    <b>MAE:</b> 0.2620<br>
                    <b>RMSE:</b> 0.3542<br>
                    <b>R²:</b> 0.5942
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>🧠 Cuisine Classification</h3>
                <p>Improved classifier evaluation on the selected 20 cuisine classes</p>
                <p style="font-size:1rem;line-height:2;margin-top:12px;">
                    <b>Accuracy:</b> 48.88%<br>
                    <b>Weighted F1:</b> 0.5061<br>
                    <b>Classes:</b> 20
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "Model predictions are estimates, not verified restaurant facts. "
        "Cuisine classification is limited to the classes represented in its training data."
    )

    st.markdown(
        """
        <div class="hero" style="margin-top:24px;margin-bottom:0;">
            <div class="hero-eyebrow">CREATED BY</div>
            <h1 style="font-size:2rem;">Harshita Mishra</h1>
            <p>Machine Learning Developer · Restaurant Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------- FOOTER -----------------------------

st.markdown(
    f"""
    <div class="footer">
        <b>Restaurant Intelligence</b> &nbsp;·&nbsp; Harshita Mishra
        &nbsp;·&nbsp; Machine Learning Developer
        <br>Predictions are estimates. Dataset-derived insights may reflect sampling limitations.
    </div>
    """,
    unsafe_allow_html=True,
)