import os

from .data_processing import SESSION_AI_CHAT_HISTORY_KEY


class OpenAIChatConfigurationError(Exception):
    pass


def get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise OpenAIChatConfigurationError(
            'OpenAI Python SDK o‘rnatilmagan. `pip install openai` ni ishga tushiring.'
        ) from exc

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise OpenAIChatConfigurationError(
            'OPENAI_API_KEY topilmadi. Uni environment variable sifatida sozlang.'
        )
    return OpenAI(api_key=api_key)


def get_chat_model() -> str:
    return os.getenv('OPENAI_MODEL', 'gpt-5.2')


def build_dataset_context(analysis: dict, file_name: str | None) -> str:
    preview_columns = analysis.get('preview_columns', [])
    preview_rows = analysis.get('preview_rows', [])[:5]
    top_products = analysis.get('top_products', [])
    regional_analysis = analysis.get('regional_analysis', [])
    date_range = analysis.get('date_range')

    return (
        f"Current dataset file: {file_name or 'unknown'}\n"
        f"Total sales: {analysis.get('total_sales', 0)}\n"
        f"Average sale: {analysis.get('average_sale', 0)}\n"
        f"Unique products: {analysis.get('product_count', 0)}\n"
        f"Rows analyzed: {analysis.get('row_count', 0)}\n"
        f"Detected columns: {', '.join(analysis.get('available_dimensions', []))}\n"
        f"Preview columns: {preview_columns}\n"
        f"Preview rows: {preview_rows}\n"
        f"Top products: {top_products}\n"
        f"Regional analysis: {regional_analysis}\n"
        f"Date range: {date_range}\n"
    )


def build_messages(chat_history: list[dict], user_message: str) -> list[dict]:
    messages = []
    for item in chat_history[-8:]:
        role = item.get('role')
        content = item.get('content', '')
        if role in {'user', 'assistant'} and content:
            messages.append({'role': role, 'content': content})
    messages.append({'role': 'user', 'content': user_message})
    return messages


def ask_openai_about_dataset(
    *,
    analysis: dict,
    file_name: str | None,
    user_message: str,
    chat_history: list[dict],
) -> tuple[str, list[dict]]:
    client = get_openai_client()
    model = get_chat_model()
    dataset_context = build_dataset_context(analysis, file_name)

    response = client.responses.create(
        model=model,
        reasoning={'effort': 'low'},
        instructions=(
            'You are a sales data assistant inside a Django dashboard. '
            'Answer using only the uploaded dataset context and conversation. '
            'If the answer is not supported by the dataset context, say that clearly. '
            'Be concise, practical, and user-friendly. '
            'Prefer numbers and short bullet points when useful.'
        ),
        input=[
            {
                'role': 'developer',
                'content': f'Dataset context:\n{dataset_context}',
            },
            *build_messages(chat_history, user_message),
        ],
    )

    answer = (response.output_text or '').strip()
    if not answer:
        answer = 'AI javob qaytarmadi.'

    updated_history = [
        *chat_history[-8:],
        {'role': 'user', 'content': user_message},
        {'role': 'assistant', 'content': answer},
    ]
    return answer, updated_history


def reset_chat_history(session) -> None:
    session[SESSION_AI_CHAT_HISTORY_KEY] = []
    session.modified = True
