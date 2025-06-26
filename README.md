# ElleVet Customer Churn Dashboard

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git

### macOS/Linux Setup

```bash
# Clone repository
git clone https://github.com/ehringhaus/ellevet_dashboard.git
cd ellevet_dashboard

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# IMPORTANT: Data files are NOT included in the git repository
# You must obtain the data files separately and add them to data_fast/
# Required files: customers_redacted.pkl, orders_redacted.pkl, orders_with_utm.pkl,
# quizzes_redacted.pkl, refunds_affiliated.pkl, subscriptions_redacted.pkl, tickets_redacted.pkl

# If you have CSV files instead of pickle files, use reduce_data.py to convert them:
# python3 reduce_data.py  # This converts CSV files from data/ to pickle files in data_fast/

# Run dashboard
python3 app.py
```

### Windows Setup

```cmd
# Clone repository
git clone https://github.com/your-username/ellevet_dashboard.git
cd ellevet_dashboard

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# IMPORTANT: Data files are NOT included in the git repository
# You must obtain the data files separately and add them to data_fast/
# Required files: customers_redacted.pkl, orders_redacted.pkl, orders_with_utm.pkl,
# quizzes_redacted.pkl, refunds_affiliated.pkl, subscriptions_redacted.pkl, tickets_redacted.pkl

# If you have CSV files instead of pickle files, use reduce_data.py to convert them:
# python reduce_data.py  # This converts CSV files from data/ to pickle files in data_fast/

# Run dashboard
python app.py
```

### Data Files

**IMPORTANT NOTE:** Data files are NOT included in the git repository and must be obtained separately due to their size and privacy considerations.

Create a `data_fast/` directory and add these files:
- `customers_redacted.pkl`
- `orders_redacted.pkl`
- `orders_with_utm.pkl`
- `quizzes_redacted.pkl`
- `refunds_affiliated.pkl`
- `subscriptions_redacted.pkl`
- `tickets_redacted.pkl`

Alternatively, use CSV files in a `data/` directory with the same names (`.csv` extension).

#### About reduce_data.py

The `reduce_data.py` script is a utility that converts CSV files into optimized pickle (.pkl) files. This significantly improves dashboard loading times by:
- Converting large CSV files from the `data/` directory into compressed pickle files
- Storing the optimized files in the `data_fast/` directory
- Reducing memory usage and improving performance

This will process all CSV files in the `data/` directory and create corresponding `.pkl` files in `data_fast/`. Use this script whenever you have updated CSV data files that need to be converted to the faster pickle format.

### Access Dashboard

Open browser to: `http://localhost:8050`

### Common Issues

**Module not found errors:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

**Port already in use:**
```bash
# Kill process on port 8050
lsof -ti:8050 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :8050   # Windows
```

**Memory issues:** Close other applications, ensure 8GB+ RAM available