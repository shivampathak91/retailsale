
# Retail AI SaaS

An intelligent analytics platform for retail businesses built with Streamlit, featuring AI-powered insights, forecasting, and interactive dashboards.

## Features

- **User Authentication**: Secure login/signup with Supabase
- **Data Upload & Analysis**: Upload CSV files and analyze retail sales data
- **AI-Powered Insights**: OpenAI integration for intelligent data interpretation
- **Time Series Forecasting**: Prophet-based sales forecasting
- **Customer Segmentation**: KMeans clustering for customer grouping
- **Interactive Dashboards**: Real-time visualizations with Plotly
- **PDF Reports**: Generate downloadable analysis reports
- **Supabase Backend**: Cloud database for data persistence

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: Supabase (PostgreSQL)
- **AI/ML**: 
  - OpenAI (GPT integration)
  - Prophet (Time series forecasting)
  - Scikit-learn (Machine learning algorithms)
- **Visualization**: Plotly Express
- **PDF Generation**: ReportLab

## Prerequisites

- Python 3.8+
- pip or conda
- API Keys:
  - Supabase (URL & Key)
  - OpenAI (API Key)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd retailsales
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure secrets**
   Create `.streamlit/secrets.toml`:
   ```toml
   SUPABASE_URL = "your_supabase_url"
   SUPABASE_KEY = "your_supabase_key"
   OPENAI_API_KEY = "your_openai_api_key"
   ```

## Usage

Run the application:
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## Project Structure

```
retailsales/
├── app.py              # Main Streamlit application with auth & AI features
├── app1.py             # Alternative analytics dashboard
├── requirements.txt    # Python dependencies
└── .streamlit/
    └── secrets.toml    # API configuration (keep secret)
```

## Key Components

### Authentication
- Email/password signup and login
- Session management with Supabase Auth

### Data Analysis
- CSV file upload
- Exploratory data analysis
- Statistical summaries

### Forecasting
- Time series analysis with Prophet
- Sales predictions and trend analysis

### Customer Insights
- Clustering analysis with KMeans
- Customer segmentation
- Behavioral patterns

### Visualizations
- Interactive charts with Plotly
- Dashboard-style layouts
- Real-time data exploration

## Environment Variables

Set these in secrets.toml or as environment variables:

- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase API key
- `OPENAI_API_KEY`: Your OpenAI API key

## License

[Add your license here]

## Support

For issues or questions, please create an issue in the repository.

## Future Enhancements

- Real-time data streaming
- Advanced ML models
- Export to multiple formats
- Mobile app
- API endpoints
```

You can copy this content and create a `README.md` file in your project root directory. Customize sections like License, Support contact, and add any additional information specific to your use case.You can copy this content and create a `README.md` file in your project root directory. Customize sections like License, Support contact, and add any additional information specific to your use case.
