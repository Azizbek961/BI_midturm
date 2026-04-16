from django import forms

from .models import FileUpload


class FileUploadForm(forms.ModelForm):
    class Meta:
        model = FileUpload
        fields = ['file']
        widgets = {
            'file': forms.ClearableFileInput(attrs={'class': 'form-control'}),
        }

    def clean_file(self):
        uploaded_file = self.cleaned_data['file']
        allowed_extensions = ('.csv', '.xlsx')
        max_size = 5 * 1024 * 1024

        if not uploaded_file.name.lower().endswith(allowed_extensions):
            raise forms.ValidationError('Only CSV and XLSX files are allowed.')

        if uploaded_file.size > max_size:
            raise forms.ValidationError('File size must be 5 MB or less.')

        return uploaded_file
