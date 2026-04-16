# Sales Data Analyzer Agent

A Django web application for uploading CSV or Excel sales files, previewing data, generating analytics, rendering charts, and running simple natural-language queries with Pandas.

## Features

- Upload `.csv` and `.xlsx` files
- Validate file type and enforce a 5 MB size limit
- Persist uploaded dataset in Django session
- View uploaded data in an HTML table
- Generate summary metrics:
  - Total sales
  - Number of products
  - Top-selling products
  - Regional analysis
- Render Chart.js visualizations:
  - Bar chart for top products
  - Line chart for sales over time
- Run simple text queries on the same dashboard page
- Chat with a real OpenAI-powered AI assistant inside the dashboard

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install django pandas openpyxl openai
```

3. Set your OpenAI API key as an environment variable.

PowerShell:

```powershell
$env:OPENAI_API_KEY="your_new_key_here"
```

Optional model override:

```powershell
$env:OPENAI_MODEL="gpt-5.2"
```

4. Apply migrations:

```bash
python manage.py migrate
```

5. Start the development server:

```bash
python manage.py runserver
```

6. Open `http://127.0.0.1:8000/`

## Expected dataset columns

The app automatically tries to detect these columns by name:

- Sales: `sales`, `revenue`, `amount`, `total`, `price`
- Product: `product`, `item`, `sku`, `name`
- Region: `region`, `area`, `location`, `territory`, `state`
- Date: `date`, `day`, `month`, `order`

Using clear column names such as `Product`, `Region`, `Sales`, and `Order Date` will give the best results.
