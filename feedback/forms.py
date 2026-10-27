from django import forms

class ReviewAnalysisForm(forms.Form):
    """
    Form for single review sentiment inference.
    """
    text = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 4,
            'placeholder': 'Paste customer feedback, product review, or support comment here (e.g., "The battery life exceeded all my expectations, definitely worth the price!")...'
        }),
        label="Customer Review",
        help_text="Enter customer feedback text (at least 5 characters)."
    )
    product_name = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Optional product or service name (e.g. Wireless Headphones v2)'
        }),
        label="Product / Service"
    )

    def clean_text(self):
        text = self.cleaned_data.get('text', '').strip()
        if len(text) < 5:
            raise forms.ValidationError("Please provide at least 5 characters for meaningful sentiment analysis.")
        return text


class CSVUploadForm(forms.Form):
    """
    Form for bulk CSV customer review upload.
    """
    csv_file = forms.FileField(
        label="Select CSV File",
        help_text="Upload a CSV file (max 10MB). File must contain a column named 'review', 'text', 'content', or 'review_text'.",
        widget=forms.FileInput(attrs={'class': 'form-control', 'accept': '.csv'})
    )

    def clean_csv_file(self):
        file = self.cleaned_data.get('csv_file')
        if not file:
            raise forms.ValidationError("No file was uploaded.")

        if not file.name.lower().endswith('.csv'):
            raise forms.ValidationError("Only valid CSV (.csv) files are supported.")

        # Limit file size to 10 MB
        if file.size > 10 * 1024 * 1024:
            raise forms.ValidationError("File size exceeds 10MB limit. Please upload a smaller dataset.")

        return file
