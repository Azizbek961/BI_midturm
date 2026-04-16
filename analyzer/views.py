from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST

from .data_processing import (
    SESSION_ANALYSIS_KEY,
    SESSION_AI_CHAT_HISTORY_KEY,
    SESSION_DATA_KEY,
    SESSION_FILE_NAME_KEY,
    answer_query,
    build_analysis,
    deserialize_dataframe,
    load_dataframe,
    prepare_dataframe,
    sanitize_for_json,
    serialize_dataframe,
)
from .forms import FileUploadForm
from .openai_chat import (
    OpenAIChatConfigurationError,
    ask_openai_about_dataset,
    reset_chat_history,
)


def upload_file_view(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.save()
            try:
                dataframe = load_dataframe(upload.file)
                dataframe, detected_columns = prepare_dataframe(dataframe)
                if dataframe.empty:
                    upload.delete()
                    messages.error(request, 'The uploaded file is empty.')
                    return redirect('upload')

                analysis = build_analysis(dataframe, detected_columns)
                request.session[SESSION_DATA_KEY] = serialize_dataframe(dataframe)
                request.session[SESSION_ANALYSIS_KEY] = sanitize_for_json(analysis)
                request.session[SESSION_FILE_NAME_KEY] = upload.file.name.split('/')[-1]
                reset_chat_history(request.session)
                request.session.modified = True
                return redirect('dashboard')
            except Exception as exc:
                upload.delete()
                messages.error(request, f'Unable to process the file: {exc}')
                return redirect('upload')
    else:
        form = FileUploadForm()

    return render(request, 'upload.html', {'form': form})


@require_GET
def analyze_data_view(request):
    dataset = request.session.get(SESSION_DATA_KEY)
    analysis = request.session.get(SESSION_ANALYSIS_KEY)
    file_name = request.session.get(SESSION_FILE_NAME_KEY)

    if not dataset or not analysis:
        messages.info(request, 'Upload a CSV or Excel file to start analysis.')
        return redirect('upload')

    return render(
        request,
        'dashboard.html',
        {
            'file_name': file_name,
            'analysis': analysis,
        },
    )


@require_POST
def query_data_view(request):
    dataset = request.session.get(SESSION_DATA_KEY)
    analysis = request.session.get(SESSION_ANALYSIS_KEY)
    query = request.POST.get('query', '')

    if not dataset or not analysis:
        return JsonResponse({'error': 'No dataset is available. Upload a file first.'}, status=400)

    dataframe = deserialize_dataframe(dataset)
    result = answer_query(dataframe, analysis, query)
    return JsonResponse({'result': sanitize_for_json(result)})


@require_POST
def ai_chat_view(request):
    dataset = request.session.get(SESSION_DATA_KEY)
    analysis = request.session.get(SESSION_ANALYSIS_KEY)
    file_name = request.session.get(SESSION_FILE_NAME_KEY)
    chat_history = request.session.get(SESSION_AI_CHAT_HISTORY_KEY, [])
    message = request.POST.get('message', '')

    if not dataset or not analysis:
        return JsonResponse({'error': 'No dataset is available. Upload a file first.'}, status=400)

    try:
        answer, updated_history = ask_openai_about_dataset(
            analysis=analysis,
            file_name=file_name,
            user_message=message,
            chat_history=chat_history,
        )
        request.session[SESSION_AI_CHAT_HISTORY_KEY] = updated_history
        request.session.modified = True
        return JsonResponse({'result': {'type': 'text', 'message': answer}})
    except OpenAIChatConfigurationError as exc:
        return JsonResponse(
            {
                'error': (
                    f'{exc} Windows PowerShell uchun misol: '
                    '$env:OPENAI_API_KEY="your_new_key"'
                )
            },
            status=500,
        )
    except Exception as exc:
        return JsonResponse({'error': f'OpenAI chat error: {exc}'}, status=500)
