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
            "If the dataset does not support the answer, say so clearly. "
            "Keep responses short and practical."
        ),
        input=[
            {"role": "developer", "content": f"Dataset context:\n{dataset_context}"},
            *messages,
        ],
    )
    return (response.output_text or "").strip() or "AI javob qaytarmadi."


def metric_card(title, value, caption):
    st.markdown(
        f"""
        <div style="padding:18px;border:1px solid rgba(15,23,42,.08);border-radius:18px;
        background:linear-gradient(135deg,#ffffff 0%,#f8fbff 100%);box-shadow:0 10px 24px rgba(15,23,42,.06);">
            <div style="font-size:12px;text-transform:uppercase;color:#64748b;">{title}</div>
            <div style="font-size:30px;font-weight:700;color:#0f172a;margin-top:6px;">{value}</div>
            <div style="font-size:13px;color:#64748b;margin-top:6px;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f6f1e8 0%, #ffffff 45%, #eef5f7 100%);
        }
        .hero-box {
            padding: 24px;
            border-radius: 22px;
            color: white;
            background:
                radial-gradient(circle at top right, rgba(30,64,175,.18), transparent 30%),
                radial-gradient(circle at bottom left, rgba(15,118,110,.20), transparent 28%),
                linear-gradient(135deg, #0f172a 0%, #15314b 52%, #0f766e 100%);
            margin-bottom: 18px;
        }
        .small-soft {
            color: rgba(255,255,255,.75);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-box">
        <div style="font-size:12px;text-transform:uppercase;letter-spacing:.08em;">Streamlit Deployment Version</div>
        <div style="font-size:38px;font-weight:800;margin-top:6px;">Sales Data Analyzer Agent</div>
        <div class="small-soft" style="margin-top:8px;">Upload CSV yoki XLSX fayl, analiz qiling va AI bilan chat qiling.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("CSV yoki XLSX file yuklang", type=["csv", "xlsx"])
    use_real_ai = st.toggle("OpenAI chat ishlatilsin", value=True)
    st.caption("OPENAI_API_KEY environment variable orqali olinadi.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is None:
    st.info("Ishni boshlash uchun fayl yuklang.")
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

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Total sales", f"{analysis['total_sales']:,.2f}", "Umumiy revenue")
with col2:
    metric_card("Average sale", f"{analysis['average_sale']:,.2f}", "Qator bo'yicha o'rtacha")
with col3:
    metric_card("Unique products", f"{analysis['product_count']:,}", "Takrorlanmas mahsulotlar")
with col4:
    metric_card("Rows analyzed", f"{analysis['row_count']:,}", "Tahlil qilingan yozuvlar")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Quick insights")
    st.write(f"Fayl: `{uploaded_file.name}`")
    st.write(f"Detected dimensions: {', '.join(analysis['available_dimensions']) or 'Aniqlanmadi'}")
    if analysis["top_product"]:
        st.success(f"Top product: {analysis['top_product']['label']} ({analysis['top_product']['value']:,.2f})")
    if analysis["top_region"]:
        st.info(f"Top region: {analysis['top_region']['label']} ({analysis['top_region']['value']:,.2f})")
    if analysis["date_range"]:
        st.caption(f"Date range: {analysis['date_range']['start']} to {analysis['date_range']['end']}")

    st.subheader("Quick query")
    quick_query = st.text_input("Savol yozing", placeholder="Top 5 products")
    if st.button("Query ishlatish", use_container_width=True):
        result = simple_query(df, analysis, quick_query)
        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
        else:
            st.write(result)

with right_col:
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("Top products")
        if analysis["top_products"].empty:
            st.warning("Bar chart uchun kerakli ustunlar topilmadi.")
        else:
            top_chart_df = analysis["top_products"].set_index("label")[["value"]]
            st.bar_chart(top_chart_df, height=320)

    with chart_right:
        st.subheader("Sales over time")
        if analysis["sales_over_time"].empty:
            st.warning("Line chart uchun date va sales ustunlari kerak.")
        else:
            time_chart_df = analysis["sales_over_time"].set_index("period")[["sales"]]
            st.line_chart(time_chart_df, height=320)

st.subheader("Regional analysis")
if analysis["regional_analysis"].empty:
    st.warning("Region analysis mavjud emas.")
else:
    st.dataframe(analysis["regional_analysis"], use_container_width=True)

st.subheader("Data preview")
st.dataframe(analysis["preview"], use_container_width=True)

st.subheader("AI bilan chat")
st.caption("Dataset context asosida javob beradi.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_message = st.chat_input("Savolingizni yozing")
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
                    st.dataframe(answer, use_container_width=True)
                    answer = "Yuqorida jadval ko'rinishida chiqarildi."
            st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
