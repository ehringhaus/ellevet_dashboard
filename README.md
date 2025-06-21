# ElleVet Customer Churn Dashboard

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git

### macOS/Linux Setup

```bash
# Clone repository
git clone https://github.com/your-username/ellevet_dashboard.git
cd ellevet_dashboard

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Add data files to data_fast/ directory (see Data Files section)

# Run dashboard
python app.py
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

# Add data files to data_fast/ directory (see Data Files section)

# Run dashboard
python app.py
```

### Data Files

Create a `data_fast/` directory and add these files:
- `customers_redacted.pkl`
- `orders_redacted.pkl`
- `orders_with_utm.pkl`
- `quizzes_redacted.pkl`
- `refunds_affiliated.pkl`
- `subscriptions_redacted.pkl`
- `tickets_redacted.pkl`

Alternatively, use CSV files in a `data/` directory with the same names (`.csv` extension).

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