import os

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


st.set_page_config(
    page_title="Sales Data Analyzer Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def find_column(columns, keywords):
    normalized = {str(column).strip().lower(): column for column in columns}
    for keyword in keywords:
        for normalized_name, original_name in normalized.items():
            if keyword in normalized_name:
                return original_name
    return None


def load_dataframe(uploaded_file):
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Only CSV and XLSX files are supported.")


def prepare_dataframe(dataframe):
    df = dataframe.copy()
    df.columns = [str(column).strip() for column in df.columns]
    df = df.dropna(how="all")

    detected_columns = {
        "sales": find_column(df.columns, ["sales", "revenue", "amount", "total", "price"]),
        "product": find_column(df.columns, ["product", "item", "sku", "name"]),
        "region": find_column(df.columns, ["region", "area", "location", "territory", "state"]),
        "date": find_column(df.columns, ["date", "day", "month", "order"]),
    }

    sales_column = detected_columns["sales"]
    if sales_column:
        cleaned_sales = (
            df[sales_column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        df[sales_column] = pd.to_numeric(cleaned_sales, errors="coerce").fillna(0)

    date_column = detected_columns["date"]
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    return df, detected_columns


def build_analysis(df, detected_columns):
    sales_column = detected_columns.get("sales")
    product_column = detected_columns.get("product")
    region_column = detected_columns.get("region")
    date_column = detected_columns.get("date")

    total_sales = float(df[sales_column].sum()) if sales_column and sales_column in df.columns else 0.0
    average_sale = float(df[sales_column].mean()) if sales_column and sales_column in df.columns else 0.0
    product_count = int(df[product_column].dropna().nunique()) if product_column and product_column in df.columns else 0

    top_products = pd.DataFrame(columns=["label", "value"])
    if sales_column and product_column and sales_column in df.columns and product_column in df.columns:
        top_products = (
            df.groupby(product_column, dropna=True)[sales_column]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_products.columns = ["label", "value"]

    regional_analysis = pd.DataFrame(columns=["label", "value"])
    if sales_column and region_column and sales_column in df.columns and region_column in df.columns:
        regional_analysis = (
            df.groupby(region_column, dropna=True)[sales_column]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        regional_analysis.columns = ["label", "value"]

    sales_over_time = pd.DataFrame(columns=["period", "sales"])
    if sales_column and date_column and sales_column in df.columns and date_column in df.columns:
        dated_df = df.dropna(subset=[date_column]).copy()
        if not dated_df.empty:
            dated_df["period"] = pd.to_datetime(dated_df[date_column]).dt.strftime("%Y-%m-%d")
            sales_over_time = dated_df.groupby("period")[sales_column].sum().reset_index()
            sales_over_time.columns = ["period", "sales"]

    date_range = None
    if not sales_over_time.empty:
        date_range = {
            "start": sales_over_time["period"].iloc[0],
            "end": sales_over_time["period"].iloc[-1],
        }

    return {
        "total_sales": round(total_sales, 2),
        "average_sale": round(average_sale, 2),
        "product_count": product_count,
        "row_count": int(len(df.index)),
        "top_products": top_products,
        "regional_analysis": regional_analysis,
        "sales_over_time": sales_over_time,
        "preview": df.head(10),
        "date_range": date_range,
        "top_product": top_products.iloc[0].to_dict() if not top_products.empty else None,
        "top_region": regional_analysis.iloc[0].to_dict() if not regional_analysis.empty else None,
        "available_dimensions": [label for label, column in detected_columns.items() if column],
        "detected_columns": detected_columns,
    }


def simple_query(df, analysis, query):
    text = query.strip().lower()
    if not text:
        return "Savol yozing."

    if "top" in text:
        top_n = 5
        digits = "".join(character for character in text if character.isdigit())
        if digits:
            top_n = max(1, int(digits))
        top_products = analysis["top_products"].head(top_n)
        if top_products.empty:
            return "Top products uchun kerakli ustunlar topilmadi."
        return top_products

    if "region" in text:
        if analysis["regional_analysis"].empty:
            return "Region analysis mavjud emas."
        return analysis["regional_analysis"]

    if "total" in text or "revenue" in text or "sales" in text:
        return f"Total sales: {analysis['total_sales']:,.2f}"

    if "average" in text:
        return f"Average sale: {analysis['average_sale']:,.2f}"

    if "product" in text:
        return f"Unique products: {analysis['product_count']}"

    return "Savol tanilmadi. Masalan: Top 5 products, Sales by region, Total revenue."


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY topilmadi.")
    if OpenAI is None:
        raise RuntimeError("`openai` paketi o'rnatilmagan.")
    return OpenAI(api_key=api_key)


def ask_openai(analysis, file_name, question, history):
    client = get_openai_client()
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")

    dataset_context = (
        f"File: {file_name}\n"
        f"Total sales: {analysis['total_sales']}\n"
        f"Average sale: {analysis['average_sale']}\n"
        f"Product count: {analysis['product_count']}\n"
        f"Row count: {analysis['row_count']}\n"
        f"Detected dimensions: {analysis['available_dimensions']}\n"
        f"Top products: {analysis['top_products'].head(10).to_dict(orient='records')}\n"
        f"Regional analysis: {analysis['regional_analysis'].to_dict(orient='records')}\n"
        f"Sales over time: {analysis['sales_over_time'].head(30).to_dict(orient='records')}\n"
        f"Preview rows: {analysis['preview'].to_dict(orient='records')}\n"
    )

    messages = []
    for item in history[-8:]:
        if item["role"] in {"user", "assistant"}:
            messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": question})

    response = client.responses.create(
        model=model,
        reasoning={"effort": "low"},
        instructions=(
            "You are a sales data analysis assistant inside a Streamlit app. "
            "Answer only from the provided dataset context and chat history. "
            "If the dataset does not support the answer, say that clearly. "
            "Keep responses short and practical."
        ),
        input=[
            {"role": "developer", "content": f"Dataset context:\n{dataset_context}"},
            *messages,
        ],
    )
    return (response.output_text or "").strip() or "AI javob qaytarmadi."


def inject_styles():
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(14, 116, 144, 0.08), transparent 22%),
                    linear-gradient(180deg, #f5efe4 0%, #ffffff 42%, #edf6f7 100%);
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f172a 0%, #10263d 100%);
                border-right: 1px solid rgba(255,255,255,.08);
            }
            [data-testid="stSidebar"] * {
                color: #f8fafc;
            }
            .hero-shell {
                padding: 1.6rem 1.8rem;
                border-radius: 24px;
                color: white;
                background:
                    radial-gradient(circle at top right, rgba(59, 130, 246, .24), transparent 28%),
                    radial-gradient(circle at bottom left, rgba(16, 185, 129, .18), transparent 24%),
                    linear-gradient(135deg, #0f172a 0%, #15314b 52%, #0f766e 100%);
                box-shadow: 0 24px 50px rgba(15, 23, 42, 0.16);
                margin-bottom: 1rem;
            }
            .hero-kicker {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                opacity: .75;
            }
            .hero-title {
                font-size: 2.35rem;
                font-weight: 800;
                line-height: 1.05;
                margin: .5rem 0;
            }
            .hero-soft {
                color: rgba(255,255,255,.76);
                max-width: 760px;
            }
            .metric-box {
                padding: 1rem 1.1rem;
                border: 1px solid rgba(15,23,42,.08);
                border-radius: 20px;
                background: linear-gradient(135deg,#ffffff 0%,#f8fbff 100%);
                box-shadow: 0 12px 24px rgba(15,23,42,.06);
            }
            .metric-title {
                font-size: 0.74rem;
                text-transform: uppercase;
                letter-spacing: .08em;
                color: #64748b;
            }
            .metric-value {
                font-size: 2rem;
                font-weight: 800;
                color: #0f172a;
                margin-top: .35rem;
            }
            .metric-caption {
                color: #64748b;
                font-size: .82rem;
                margin-top: .4rem;
            }
            .section-card {
                background: rgba(255,255,255,.88);
                border: 1px solid rgba(15,23,42,.08);
                border-radius: 22px;
                padding: 1rem;
                box-shadow: 0 14px 30px rgba(15,23,42,.06);
            }
            .mini-card {
                padding: .9rem 1rem;
                border-radius: 16px;
                border: 1px solid rgba(15,23,42,.08);
                background: #fff;
            }
            .mini-label {
                font-size: .72rem;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: .08em;
                margin-bottom: .35rem;
            }
            .pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: .45rem;
            }
            .pill {
                padding: .42rem .72rem;
                border-radius: 999px;
                background: #eff6ff;
                color: #1d4ed8;
                font-size: .82rem;
                border: 1px solid rgba(29,78,216,.10);
            }
            .empty-box {
                border: 1px dashed rgba(100,116,139,.4);
                border-radius: 18px;
                padding: 2rem 1rem;
                color: #64748b;
                text-align: center;
                background: linear-gradient(180deg,#f8fafc 0%,#ffffff 100%);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(uploaded_file=None, analysis=None):
    file_line = ""
    if uploaded_file and analysis:
        file_line = (
            f"<div class='hero-soft' style='margin-top:.65rem;'>"
            f"File: <strong>{uploaded_file.name}</strong> · Rows: <strong>{analysis['row_count']:,}</strong> · "
            f"Detected: <strong>{', '.join(analysis['available_dimensions']) or 'none'}</strong>"
            f"</div>"
        )
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">Streamlit deployment version</div>
            <div class="hero-title">Sales Data Analyzer Agent</div>
            <div class="hero-soft">Upload CSV yoki XLSX fayl, dashboardni ko‘ring, quick query ishlating va AI bilan dataset ustida chat qiling.</div>
            {file_line}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title, value, caption):
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_open(title, subtitle=""):
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def section_close():
    st.markdown("</div>", unsafe_allow_html=True)


def render_empty(message):
    st.markdown(f"<div class='empty-box'>{message}</div>", unsafe_allow_html=True)


def run_quick_query(df, analysis, query):
    result = simple_query(df, analysis, query)
    if isinstance(result, pd.DataFrame):
        st.dataframe(result, use_container_width=True, hide_index=True)
    else:
        st.info(result)


inject_styles()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "quick_query_value" not in st.session_state:
    st.session_state.quick_query_value = ""

with st.sidebar:
    st.markdown("## Control Panel")
    st.caption("Data upload, AI mode va tezkor sozlamalar shu yerda.")
    uploaded_file = st.file_uploader("CSV yoki XLSX file yuklang", type=["csv", "xlsx"])
    use_real_ai = st.toggle("OpenAI chat ishlatilsin", value=True)
    st.caption("OPENAI_API_KEY environment variable orqali olinadi.")
    st.divider()
    st.markdown("### Tavsiya")
    st.caption("Aniqroq natija uchun `Product`, `Region`, `Sales`, `Order Date` ustunlaridan foydalaning.")

render_hero()

if uploaded_file is None:
    st.markdown(
        """
        <div class="section-card">
            <h3 style="margin-top:0;">Start here</h3>
            <p style="color:#64748b;">Chap paneldan fayl yuklang. Ilova avtomatik ravishda KPI, charts, region analysis va AI chatni tayyorlaydi.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    raw_df = load_dataframe(uploaded_file)
    df, detected_columns = prepare_dataframe(raw_df)
except Exception as exc:
    st.error(f"Faylni o'qib bo'lmadi: {exc}")
    st.stop()

if df.empty:
    st.warning("Yuklangan fayl bo'sh.")
    st.stop()

analysis = build_analysis(df, detected_columns)
render_hero(uploaded_file, analysis)

metric_cols = st.columns(4)
with metric_cols[0]:
    metric_card("Total sales", f"{analysis['total_sales']:,.2f}", "Umumiy revenue")
with metric_cols[1]:
    metric_card("Average sale", f"{analysis['average_sale']:,.2f}", "Bir yozuv bo'yicha o'rtacha")
with metric_cols[2]:
    metric_card("Unique products", f"{analysis['product_count']:,}", "Takrorlanmas mahsulotlar")
with metric_cols[3]:
    metric_card("Rows analyzed", f"{analysis['row_count']:,}", "Tahlil qilingan qatorlar")

tab_overview, tab_charts, tab_data, tab_chat = st.tabs(["Overview", "Charts", "Data", "AI Chat"])

with tab_overview:
    left_col, right_col = st.columns([1, 1.45], gap="large")

    with left_col:
        section_open("Quick insights", "Dataset bo‘yicha eng muhim signal va tezkor savollar.")
        insight_cols = st.columns(2)
        with insight_cols[0]:
            st.markdown("<div class='mini-card'>", unsafe_allow_html=True)
            st.markdown("<div class='mini-label'>Top product</div>", unsafe_allow_html=True)
            if analysis["top_product"]:
                st.markdown(f"**{analysis['top_product']['label']}**")
                st.caption(f"{analysis['top_product']['value']:,.2f} sales")
            else:
                st.caption("Aniqlanmadi")
            st.markdown("</div>", unsafe_allow_html=True)
        with insight_cols[1]:
            st.markdown("<div class='mini-card'>", unsafe_allow_html=True)
            st.markdown("<div class='mini-label'>Top region</div>", unsafe_allow_html=True)
            if analysis["top_region"]:
                st.markdown(f"**{analysis['top_region']['label']}**")
                st.caption(f"{analysis['top_region']['value']:,.2f} sales")
            else:
                st.caption("Aniqlanmadi")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='mini-card' style='margin-top:.85rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='mini-label'>Detected dimensions</div>", unsafe_allow_html=True)
        if analysis["available_dimensions"]:
            pills = "".join(f"<span class='pill'>{item}</span>" for item in analysis["available_dimensions"])
            st.markdown(f"<div class='pill-row'>{pills}</div>", unsafe_allow_html=True)
        else:
            st.caption("Hech narsa aniqlanmadi.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='mini-card' style='margin-top:.85rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='mini-label'>Quick query</div>", unsafe_allow_html=True)
        button_cols = st.columns(2)
        quick_options = [
            "Top 5 products",
            "Sales by region",
            "Total revenue",
            "Average sale",
        ]
        for index, option in enumerate(quick_options):
            with button_cols[index % 2]:
                if st.button(option, use_container_width=True, key=f"quick-{option}"):
                    st.session_state.quick_query_value = option

        quick_query = st.text_input(
            "Savol yozing",
            value=st.session_state.quick_query_value,
            placeholder="Masalan: Top 5 products",
        )
        if st.button("Run quick query", use_container_width=True):
            run_quick_query(df, analysis, quick_query)
            st.session_state.quick_query_value = quick_query
        st.markdown("</div>", unsafe_allow_html=True)
        section_close()

    with right_col:
        section_open("Regional analysis", "Hududlar kesimidagi sotuvlar va vaqt oralig‘i.")
        info_cols = st.columns(2)
        with info_cols[0]:
            st.metric("File name", uploaded_file.name)
        with info_cols[1]:
            if analysis["date_range"]:
                st.metric("Date range", f"{analysis['date_range']['start']} -> {analysis['date_range']['end']}")
            else:
                st.metric("Date range", "Not detected")

        if analysis["regional_analysis"].empty:
            render_empty("Region analysis uchun kerakli ustunlar topilmadi.")
        else:
            st.dataframe(
                analysis["regional_analysis"],
                use_container_width=True,
                hide_index=True,
            )
        section_close()

with tab_charts:
    chart_left, chart_right = st.columns(2, gap="large")

    with chart_left:
        section_open("Top products", "Eng ko‘p sotilgan mahsulotlar.")
        if analysis["top_products"].empty:
            render_empty("Bar chart uchun product va sales ustunlari kerak.")
        else:
            top_chart_df = analysis["top_products"].set_index("label")[["value"]]
            st.bar_chart(top_chart_df, height=340)
            st.dataframe(analysis["top_products"].head(5), use_container_width=True, hide_index=True)
        section_close()

    with chart_right:
        section_open("Sales over time", "Vaqt bo‘yicha sales trend.")
        if analysis["sales_over_time"].empty:
            render_empty("Line chart uchun date va sales ustunlari kerak.")
        else:
            time_chart_df = analysis["sales_over_time"].set_index("period")[["sales"]]
            st.line_chart(time_chart_df, height=340)
            st.dataframe(analysis["sales_over_time"].tail(5), use_container_width=True, hide_index=True)
        section_close()

with tab_data:
    top_data_left, top_data_right = st.columns([1.05, 1.25], gap="large")

    with top_data_left:
        section_open("Dataset summary", "Struktura va ustunlar haqida qisqa ma’lumot.")
        st.write(f"Rows: **{analysis['row_count']:,}**")
        st.write(f"Columns: **{len(df.columns):,}**")
        st.write("Detected columns map:")
        st.json(analysis["detected_columns"])
        section_close()

    with top_data_right:
        section_open("Data preview", "Birinchi 10 qator.")
        st.dataframe(analysis["preview"], use_container_width=True, hide_index=True)
        section_close()

with tab_chat:
    chat_left, chat_right = st.columns([1.15, 1.85], gap="large")

    with chat_left:
        section_open("AI settings", "OpenAI chat yoki local quick-query mode.")
        ai_status = "Enabled" if use_real_ai else "Local mode"
        st.metric("AI mode", ai_status)
        st.metric("Model", os.getenv("OPENAI_MODEL", "gpt-5.2") if use_real_ai else "Rule-based")
        if use_real_ai:
            st.caption("OpenAI ishlashi uchun `OPENAI_API_KEY` o‘rnatilgan bo‘lishi kerak.")
        else:
            st.caption("Bu rejimda oddiy local query logic ishlaydi.")
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        section_close()

    with chat_right:
        section_open("AI bilan chat", "Dataset context asosida javob beradi.")
        if not st.session_state.chat_history:
            st.info("Suhbatni boshlang. Masalan: Total revenue, strongest region, top 3 products.")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_message = st.chat_input("Savolingizni yozing", key="main_chat_input")
        if user_message:
            st.session_state.chat_history.append({"role": "user", "content": user_message})
            with st.chat_message("user"):
                st.markdown(user_message)

            with st.chat_message("assistant"):
                with st.spinner("Javob tayyorlanmoqda..."):
                    if use_real_ai:
                        try:
                            answer = ask_openai(
                                analysis=analysis,
                                file_name=uploaded_file.name,
                                question=user_message,
                                history=st.session_state.chat_history[:-1],
                            )
                        except Exception as exc:
                            answer = (
                                f"OpenAI chat ishlamadi: {exc}\n\n"
                                "PowerShell misol:\n"
                                '$env:OPENAI_API_KEY="your_new_key"'
                            )
                    else:
                        answer = simple_query(df, analysis, user_message)
                        if isinstance(answer, pd.DataFrame):
                            st.dataframe(answer, use_container_width=True, hide_index=True)
                            answer = "Natija yuqorida jadval ko‘rinishida chiqarildi."
                    st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        section_close()
