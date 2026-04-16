import math
from typing import Any

import pandas as pd


SESSION_DATA_KEY = 'uploaded_dataset'
SESSION_FILE_NAME_KEY = 'uploaded_filename'
SESSION_ANALYSIS_KEY = 'uploaded_analysis'
SESSION_AI_CHAT_HISTORY_KEY = 'ai_chat_history'


def load_dataframe(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    if name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    raise ValueError('Unsupported file type.')


def find_column(columns, keywords):
    normalized = {str(column).strip().lower(): column for column in columns}
    for keyword in keywords:
        for normalized_name, original_name in normalized.items():
            if keyword in normalized_name:
                return original_name
    return None


def prepare_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    df = dataframe.copy()
    df.columns = [str(column).strip() for column in df.columns]
    df = df.dropna(how='all')

    detected_columns = {
        'sales': find_column(df.columns, ['sales', 'revenue', 'amount', 'total', 'price']),
        'product': find_column(df.columns, ['product', 'item', 'sku', 'name']),
        'region': find_column(df.columns, ['region', 'area', 'location', 'territory', 'state']),
        'date': find_column(df.columns, ['date', 'day', 'month', 'order']),
    }

    sales_column = detected_columns['sales']
    if sales_column:
        cleaned_sales = (
            df[sales_column]
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('$', '', regex=False)
            .str.strip()
        )
        df[sales_column] = pd.to_numeric(cleaned_sales, errors='coerce').fillna(0)

    date_column = detected_columns['date']
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    return df, detected_columns


def serialize_dataframe(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    serializable_df = dataframe.copy()
    for column in serializable_df.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable_df[column]):
            serializable_df[column] = serializable_df[column].dt.strftime('%Y-%m-%d')
    serializable_df = serializable_df.where(pd.notnull(serializable_df), None)
    return serializable_df.to_dict(orient='records')


def deserialize_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records)


def build_analysis(df: pd.DataFrame, detected_columns: dict[str, str | None]) -> dict[str, Any]:
    sales_column = detected_columns.get('sales')
    product_column = detected_columns.get('product')
    region_column = detected_columns.get('region')
    date_column = detected_columns.get('date')

    total_sales = 0.0
    if sales_column and sales_column in df.columns:
        total_sales = float(df[sales_column].fillna(0).sum())

    product_count = 0
    if product_column and product_column in df.columns:
        product_count = int(df[product_column].dropna().nunique())

    average_sale = 0.0
    if sales_column and sales_column in df.columns and len(df.index) > 0:
        average_sale = float(df[sales_column].fillna(0).mean())

    top_products = []
    if sales_column and product_column and sales_column in df.columns and product_column in df.columns:
        grouped = (
            df.groupby(product_column, dropna=True)[sales_column]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        top_products = [
            {'label': str(index), 'value': round(float(value), 2)}
            for index, value in grouped.items()
        ]

    regional_analysis = []
    if sales_column and region_column and sales_column in df.columns and region_column in df.columns:
        grouped = (
            df.groupby(region_column, dropna=True)[sales_column]
            .sum()
            .sort_values(ascending=False)
        )
        regional_analysis = [
            {'label': str(index), 'value': round(float(value), 2)}
            for index, value in grouped.items()
        ]

    top_region = regional_analysis[0] if regional_analysis else None

    sales_over_time = []
    if sales_column and date_column and sales_column in df.columns and date_column in df.columns:
        dated = df.dropna(subset=[date_column]).copy()
        if not dated.empty:
            dated['period'] = pd.to_datetime(dated[date_column]).dt.strftime('%Y-%m-%d')
            grouped = dated.groupby('period')[sales_column].sum().sort_index()
            sales_over_time = [
                {'label': str(index), 'value': round(float(value), 2)}
                for index, value in grouped.items()
            ]

    date_range = None
    if sales_over_time:
        date_range = {
            'start': sales_over_time[0]['label'],
            'end': sales_over_time[-1]['label'],
        }

    top_product = top_products[0] if top_products else None

    preview_rows = serialize_dataframe(df.head(10))

    return {
        'total_sales': round(total_sales, 2),
        'product_count': product_count,
        'average_sale': round(average_sale, 2),
        'top_product': top_product,
        'top_region': top_region,
        'date_range': date_range,
        'top_products': top_products,
        'regional_analysis': regional_analysis,
        'sales_over_time': sales_over_time,
        'preview_columns': list(df.columns),
        'preview_rows': preview_rows,
        'row_count': int(len(df.index)),
        'detected_columns': detected_columns,
        'available_dimensions': [label for label, column in detected_columns.items() if column],
    }


def answer_query(df: pd.DataFrame, analysis: dict[str, Any], query: str) -> dict[str, Any]:
    normalized_query = query.strip().lower()
    if not normalized_query:
        return {'type': 'text', 'message': 'Enter a question such as "Top 5 products" or "Sales by region".'}

    if 'top' in normalized_query:
        top_limit = 5
        digits = ''.join(character for character in normalized_query if character.isdigit())
        if digits:
            top_limit = max(1, int(digits))

        sales_column = analysis.get('detected_columns', {}).get('sales')
        product_column = analysis.get('detected_columns', {}).get('product')
        if not sales_column or not product_column or sales_column not in df.columns or product_column not in df.columns:
            return {'type': 'text', 'message': 'Top product analysis is unavailable. Make sure the file includes product and sales columns.'}

        grouped = (
            df.groupby(product_column, dropna=True)[sales_column]
            .sum()
            .sort_values(ascending=False)
            .head(top_limit)
        )
        return {
            'type': 'list',
            'title': f'Top {top_limit} products',
            'items': [f'{index}: {round(float(value), 2)}' for index, value in grouped.items()],
        }

    if 'region' in normalized_query:
        regional_analysis = analysis.get('regional_analysis', [])
        if not regional_analysis:
            return {'type': 'text', 'message': 'Regional analysis is unavailable. Make sure the file includes region and sales columns.'}
        return {
            'type': 'table',
            'title': 'Sales by region',
            'headers': ['Region', 'Sales'],
            'rows': [[item['label'], item['value']] for item in regional_analysis],
        }

    if 'total' in normalized_query or 'revenue' in normalized_query or 'sales' in normalized_query:
        return {
            'type': 'text',
            'message': f"Total sales: {analysis.get('total_sales', 0):,.2f}",
        }

    if 'product' in normalized_query:
        return {
            'type': 'text',
            'message': f"Number of unique products: {analysis.get('product_count', 0)}",
        }

    return {
        'type': 'text',
        'message': 'Query not recognized. Try "Top 5 products", "Sales by region", or "Total revenue".',
    }


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 2)
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    return value
