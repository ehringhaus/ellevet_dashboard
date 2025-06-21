import dash
from dash import dcc, html, dash_table, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import pickle
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Import our custom modules
from data_loader import DataLoader
from model import ChurnPredictor
from config import CONFIG

# Initialize the Dash app
app = dash.Dash(__name__)

# Initialize our data loader and model
data_loader = DataLoader()
churn_predictor = ChurnPredictor()

# Persistent cache configuration
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "dashboard_data.pkl"
CACHE_DURATION_HOURS = 24  # Cache data for 24 hours

# Global cache for heavy computations
CACHED_DATA = None
CACHE_TIMESTAMP = None

def load_persistent_cache():
    """Load cached data from disk if available and recent"""
    global CACHED_DATA, CACHE_TIMESTAMP
    
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Check if cache is still valid
            cached_time = cache_data.get('timestamp')
            if cached_time and (datetime.now() - cached_time).total_seconds() < (CACHE_DURATION_HOURS * 3600):
                CACHED_DATA = cache_data.get('data')
                CACHE_TIMESTAMP = cached_time
                print(f"✅ Loaded cached data from {cached_time.strftime('%Y-%m-%d %H:%M')}")
                return True
            else:
                print("⚠️  Cached data expired, will regenerate")
        except Exception as e:
            print(f"⚠️  Error loading cache: {e}")
    
    return False

def save_persistent_cache(data):
    """Save data to persistent cache"""
    try:
        cache_data = {
            'data': data,
            'timestamp': datetime.now()
        }
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
        print("✅ Data cached to disk for faster future loading")
    except Exception as e:
        print(f"⚠️  Error saving cache: {e}")

# Load persistent cache on startup
load_persistent_cache()

# Business-focused dashboard with clear definitions
app.layout = html.Div([
    
    # Header with clear purpose
    html.Div([
        html.Div([
            html.H1("ElleVet Customer Retention Intelligence", className='main-title'),
            html.P("Identify cart customers at risk of not returning & understand what drives repeat purchases", className='subtitle')
        ], className='title-section'),
        html.Div([
            html.Div([
                html.Label("Analysis Date:", className='date-label'),
                dcc.DatePickerSingle(
                    id='analysis-date',
                    date=datetime.now().date(),
                    display_format='YYYY-MM-DD',
                    className='date-picker'
                )
            ], className='date-control'),
            html.Button("Refresh Data", id='refresh-btn', className='refresh-btn')
        ], className='header-controls')
    ], className='header'),

    # Executive Summary - Business-Focused Metrics
    html.Div([
        html.Div([
            html.H3(id='immediate-risk', className='metric-number'),
            html.P("Immediate Risk", className='metric-label'),
            html.P("90-180 days since order", className='metric-desc')
        ], className='metric-card urgent'),
        
        html.Div([
            html.H3(id='watch-list', className='metric-number'),
            html.P("Watch List", className='metric-label'), 
            html.P("60-90 days since order", className='metric-desc')
        ], className='metric-card warning'),
        
        html.Div([
            html.H3(id='revenue-opportunity', className='metric-number'),
            html.P("Revenue Opportunity", className='metric-label'),
            html.P("From at-risk customers", className='metric-desc')
        ], className='metric-card revenue'),
        
        html.Div([
            html.H3(id='repeat-rate', className='metric-number'),
            html.P("Repeat Purchase Rate", className='metric-label'),
            html.P("Customers who return", className='metric-desc')
        ], className='metric-card success')
    ], className='metrics-row'),

    # Section 1: Immediate Action Required
    html.Div([
        html.Div([
            html.H2("Customers Needing Immediate Attention", className='section-title'),
            html.P("Cart customers who haven't ordered in 90-180 days - still reachable with targeted outreach", 
                   className='section-description'),
            
            html.Div([
                html.Div([
                    html.Label("Risk Timeframe:", className='filter-label'),
                    dcc.Dropdown(
                        id='timeframe-filter',
                        options=[
                            {'label': '🔴 Immediate Risk (90-180 days)', 'value': 'immediate'},
                            {'label': '🟡 Watch List (60-90 days)', 'value': 'watch'},
                            {'label': '🟢 Recent Activity (30-60 days)', 'value': 'recent'},
                            {'label': '📊 All Active Window (30-180 days)', 'value': 'all_active'}
                        ],
                        value='immediate',
                        className='filter-dropdown'
                    )
                ], className='filter-group'),
                
                html.Div([
                    html.Label("State/Region:", className='filter-label'),
                    dcc.Dropdown(
                        id='state-filter',
                        placeholder="All States",
                        className='filter-dropdown'
                    )
                ], className='filter-group')
            ], className='filters-row')
        ], className='section-header'),
        
        html.Div([
            dash_table.DataTable(
                id='customer-table',
                columns=[
                    {'name': 'Customer ID', 'id': 'customer_id', 'type': 'text'},
                    {'name': 'State', 'id': 'state', 'type': 'text'},
                    {'name': 'Days Since Order', 'id': 'days_since_last_order', 'type': 'numeric'},
                    {'name': 'ML Risk %', 'id': 'churn_probability_percent', 'type': 'numeric', 'format': {'specifier': '.0f'}},
                    {'name': 'Behavior Score', 'id': 'jason_risk_score', 'type': 'numeric'},
                    {'name': 'Total Spent $', 'id': 'total_spent', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
                    {'name': 'Total Refunded $', 'id': 'total_refunded', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
                    {'name': 'Orders', 'id': 'total_orders', 'type': 'numeric'},
                    {'name': 'Product Type', 'id': 'product_form', 'type': 'text'}
                ],
                page_size=10,
                sort_action='native',
                filter_action='native',
                style_cell={'textAlign': 'left', 'padding': '4px 6px', 'fontSize': '12px', 'height': '32px'},
                style_header={'backgroundColor': '#f8fafc', 'fontWeight': 'bold', 'fontSize': '11px', 'height': '36px', 'padding': '4px 6px'},
                style_data_conditional=[]  # Removed highlighting for cleaner appearance
            ),
            html.Div([
                html.Button(
                    "📤 Export Top 10% High-Risk", 
                    id='export-btn', 
                    className='refresh-btn',
                    style={'fontSize': '0.8rem', 'marginTop': '1rem', 'background': '#059669'}
                ),
                html.Div(id='export-status', style={'marginTop': '0.5rem', 'fontSize': '0.8rem', 'color': '#059669'})
            ], style={'textAlign': 'center'})
        ], className='table-container')
    ], className='section'),

    # Section 2: Understanding Risk Factors  
    html.Div([
        html.H2("Why Customers Don't Return", className='section-title'),
        html.P(id='risk-analysis-description', className='section-description'),
        
        html.Div([
            # Key Risk Indicators
            html.Div([
                html.H4("Key Warning Signs", className='subsection-title'),
                html.Div(id='risk-indicators', className='indicators-content')
            ], className='indicators-box'),
            
            # Top Risk Factors (SHAP)
            html.Div([
                html.H4("Most Important Factors (AI Analysis)", className='subsection-title'),
                html.P("Machine learning model (best performer selected from multiple algorithms) identifies which customer behaviors most strongly predict non-return", 
                       className='chart-description'),
                html.Details([
                    html.Summary("🔍 AI Model Details", style={'fontSize': '0.8rem', 'fontWeight': '600', 'cursor': 'pointer', 'color': '#2563eb'}),
                    html.Div([
                        html.P("Algorithm: Model Selection (XGBoost, Random Forest, Gradient Boosting, Logistic Regression variants)", style={'fontSize': '0.8rem', 'margin': '0.2rem 0'}),
                        html.P("Target: Binary classification - customers who haven't returned after 180+ days (6 months)", style={'fontSize': '0.8rem', 'margin': '0.2rem 0'}),
                        html.P("Training Data: ~11K one-time customers filtered to recent 18-month window", style={'fontSize': '0.8rem', 'margin': '0.2rem 0'}),
                        html.P("Churn Rate: ~79% (9K churned vs 2.4K retained) - realistic business imbalance", style={'fontSize': '0.8rem', 'margin': '0.2rem 0'}),
                        html.P("Feature Count: 54 engineered features including Jason's + John's risk signals + Terri's clustering", style={'fontSize': '0.8rem', 'margin': '0.2rem 0'}),
                        html.P("Selection Process: Best single algorithm chosen via 5-fold cross-validation (ROC-AUC metric)", style={'fontSize': '0.8rem', 'margin': '0.2rem 0'}),
                        html.P("Interpretability: SHAP values computed for feature importance ranking", style={'fontSize': '0.8rem', 'margin': '0.2rem 0'})
                    ], style={'marginTop': '0.5rem', 'paddingLeft': '1rem', 'borderLeft': '2px solid #e5e7eb'})
                ], style={'marginBottom': '1rem'}),
                html.Details([
                    html.Summary("🔍 Data Leakage Prevention & Validation", style={'fontSize': '0.8rem', 'fontWeight': '600', 'cursor': 'pointer', 'color': '#2563eb'}),
                    html.Div([
                        html.P("✓ Temporal Consistency: Target defined as 180+ days without purchase (objective, time-based)", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'}),
                        html.P("✓ Feature Cutoff: Only historical data prior to churn prediction point (no future leakage)", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'}),
                        html.P("✓ Business Logic: Actionable timeframe (30-180 days) separate from training definition", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'}),
                        html.P("✓ Validation Strategy: Stratified train/test split (20% holdout) + 5-fold cross-validation", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'}),
                        html.P("✓ Feature Engineering: Pre-computed risk signals from Jason's analysis, no post-hoc data", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'}),
                        html.P("✓ Class Balance: Handles 79% churn rate with balanced algorithms (class_weight='balanced')", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'}),
                        html.P("✓ Model Versioning: Automatic retraining when churn definition changes", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'})
                    ], style={'marginTop': '0.5rem', 'paddingLeft': '1rem', 'borderLeft': '2px solid #e5e7eb'})
                ], style={'marginBottom': '1rem'}),
                html.Details([
                    html.Summary("📊 Feature Engineering Details", style={'fontSize': '0.8rem', 'fontWeight': '600', 'cursor': 'pointer', 'color': '#2563eb'}),
                    html.Div([
                        html.P("Purchase Behavior (12 features): Total orders, spend, order value stats, lifetime days, recency", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'}),
                        html.P("Demographics (6 features): Shipping state, primary payment method, acquisition channel/UTM", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'}),
                        html.P("Jason's Risk Signals (9 features): Support tickets, cancel tags, urgent flags, product issues, refunds", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'}),
                        html.P("John's Behavioral Patterns (9 features): Discount usage, spend tiers, escalation/site flags, interactions", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'}),
                        html.P("Product Analysis (1 feature): Product form preference (Chew, Softgel, Oil, Capsule, Other)", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'}),
                        html.P("Quiz Insights (2 features): Cat ownership flag, situational stress indicators", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'}),
                        html.P("Terri's Clustering (6 features): UMAP/PCA cluster assignments and coordinate embeddings", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'}),
                        html.P("Engineered Features (6 features): Order frequency, spend rate, avg days between orders, refund rate", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'}),
                        html.P("Missing Value Strategy: Domain-specific defaults (0 for counts, 'Unknown' for categories)", style={'fontSize': '0.75rem', 'margin': '0.2rem 0'})
                    ], style={'marginTop': '0.5rem', 'paddingLeft': '1rem', 'borderLeft': '2px solid #e5e7eb'})
                ], style={'marginBottom': '1rem'}),
                dcc.Graph(id='feature-importance-chart', config={'responsive': True})
            ], className='chart-box'),
            
            # Behavior Score Breakdown
            html.Div([
                html.H4("Customer Behavior Score Distribution", className='subsection-title'),
                html.P("Weighted heuristic risk score (0-8 scale) based on customer service interactions and behavioral patterns", 
                       className='chart-description'),
                html.Details([
                    html.Summary("🔍 Scoring Methodology", style={'fontSize': '0.8rem', 'fontWeight': '600', 'cursor': 'pointer', 'color': '#2563eb'}),
                    html.Div([
                        html.P("+1 point: Cancel request in support tickets (strong churn signal)", style={'fontSize': '0.8rem', 'margin': '0.2rem 0', 'color': '#dc2626'}),
                        html.P("+1 point: Product quality issues reported (satisfaction concern)", style={'fontSize': '0.8rem', 'margin': '0.2rem 0', 'color': '#ea580c'}),
                        html.P("+1 point: Urgent support tag (immediate dissatisfaction)", style={'fontSize': '0.8rem', 'margin': '0.2rem 0', 'color': '#ea580c'}),
                        html.P("+1 point: Multiple support tickets (2+ tickets - ongoing issues)", style={'fontSize': '0.8rem', 'margin': '0.2rem 0', 'color': '#ea580c'}),
                        html.P("+1 point: Has refund history (product didn't meet expectations)", style={'fontSize': '0.8rem', 'margin': '0.2rem 0', 'color': '#ea580c'}),
                        html.P("+1 point: High-risk acquisition (discount/affiliate - price-sensitive)", style={'fontSize': '0.8rem', 'margin': '0.2rem 0', 'color': '#ea580c'}),
                        html.P("+1 point: Low order value (avg order value < $40)", style={'fontSize': '0.8rem', 'margin': '0.2rem 0', 'color': '#ea580c'}),
                        html.P("+1 point: Long purchase gap (>120 days since last order)", style={'fontSize': '0.8rem', 'margin': '0.2rem 0', 'color': '#ea580c'}),
                        html.P("Total: 8 possible points maximum", style={'fontSize': '0.8rem', 'margin': '0.5rem 0 0.2rem 0', 'fontWeight': '600', 'color': '#374151'})
                    ], style={'marginTop': '0.5rem', 'paddingLeft': '1rem', 'borderLeft': '2px solid #e5e7eb'})
                ], style={'marginBottom': '1rem'}),
                html.Details([
                    html.Summary("📊 Score Interpretation", style={'fontSize': '0.8rem', 'fontWeight': '600', 'cursor': 'pointer', 'color': '#2563eb'}),
                    html.Div([
                        html.P("0-1: Low risk (satisfied customers with minimal issues)", style={'fontSize': '0.75rem', 'margin': '0.1rem 0', 'color': '#059669'}),
                        html.P("2-3: Elevated risk (some dissatisfaction signals - needs attention)", style={'fontSize': '0.75rem', 'margin': '0.1rem 0', 'color': '#ea580c'}),
                        html.P("4-5: High risk (multiple risk factors - priority intervention)", style={'fontSize': '0.75rem', 'margin': '0.1rem 0', 'color': '#dc2626'}),
                        html.P("6-8: Critical risk (severe issues - immediate action required)", style={'fontSize': '0.75rem', 'margin': '0.1rem 0', 'color': '#7f1d1d'})
                    ], style={'marginTop': '0.5rem', 'paddingLeft': '1rem', 'borderLeft': '2px solid #e5e7eb'})
                ], style={'marginBottom': '1rem'}),
                dcc.Graph(id='behavior-score-chart', config={'responsive': True})
            ], className='chart-box')
        ], className='analysis-grid')
    ], className='section'),

    # Section 3: What Makes Customers Return?
    html.Div([
        html.H2("What Makes Customers Come Back?", className='section-title'),
        html.P("Understanding successful customer patterns to replicate with at-risk customers", 
               className='section-description'),
        
        html.Div([
            # Geographic Success Patterns
            html.Div([
                html.H4("Geographic Retention Patterns", className='subsection-title'),
                html.P("Percentage of customers with 2+ orders by state (states with 5+ customers in 30-180 day window)", 
                       className='chart-description'),
                dcc.Graph(id='geographic-success-chart', config={'responsive': True})
            ], className='chart-box'),
            
            # Customer Journey Insights
            html.Div([
                html.H4("Customer Journey Segments", className='subsection-title'),
                html.P("UMAP clustering of all customers in actionable 30-180 day window based on behavior patterns", 
                       className='chart-description'),
                dcc.Graph(id='journey-segments-chart', config={'responsive': True})
            ], className='chart-box'),
            
            # Product Success Analysis
            html.Div([
                html.H4("Product Satisfaction Patterns", className='subsection-title'),
                html.P("Repeat rates by product type (customers with 10+ per product in actionable 30-180 day window)", 
                       className='chart-description'),
                dcc.Graph(id='product-success-chart', config={'responsive': True})
            ], className='chart-box')
        ], className='analysis-grid')
    ], className='section'),

    # Section 4: Actionable Recommendations
    html.Div([
        html.H2("Recommended Actions", className='section-title'),
        html.P("Data-driven recommendations for customer retention initiatives", className='section-description'),
        
        html.Div([
            html.Div([
                html.H4("High-Priority Interventions", className='subsection-title'),
                html.Div(id='priority-actions', className='actions-content')
            ], className='action-box'),
            
            html.Div([
                html.H4("Customer Segment Strategies", className='subsection-title'),
                html.Div(id='segment-strategies', className='actions-content')
            ], className='action-box'),
            
            html.Div([
                html.H4("Success Pattern Insights", className='subsection-title'),
                html.Div(id='success-patterns', className='actions-content')
            ], className='action-box')
        ], className='actions-grid')
    ], className='section'),
    
    # Hidden data stores
    html.Div(id='raw-data-store', style={'display': 'none'}),  # Heavy computation cache
    html.Div(id='processed-data-store', style={'display': 'none'})  # Date-dependent processing
    
], className='dashboard-container')

# Heavy computation callback - only runs on refresh or first load
@app.callback(
    Output('raw-data-store', 'children'),
    Input('refresh-btn', 'n_clicks'),
    prevent_initial_call=False
)
def load_raw_data(n_clicks):
    """Load and cache heavy computations (ML, clustering, etc.)"""
    global CACHED_DATA, CACHE_TIMESTAMP
    
    # Check if we have recent cached data (persistent or in-memory)
    if CACHED_DATA is not None and CACHE_TIMESTAMP is not None:
        time_diff = datetime.now() - CACHE_TIMESTAMP
        if time_diff.total_seconds() < (CACHE_DURATION_HOURS * 3600):
            print("⚡ Using cached data for speed")
            return CACHED_DATA
    
    # Prevent multiple simultaneous loads
    if hasattr(load_raw_data, '_loading') and load_raw_data._loading:
        print("⚡ Load already in progress, waiting...")
        return CACHED_DATA if CACHED_DATA else '{}'
    
    try:
        load_raw_data._loading = True
        print("🔄 Loading and processing data (this may take a moment...)")
        
        # Load customer data with all heavy computations
        print("📊 Step 1/3: Loading customer data...")
        customer_features = data_loader.load_and_process_data()
        
        # Get ML predictions
        print("🤖 Step 2/3: Running ML predictions...")
        predictions = churn_predictor.predict_churn(customer_features)
        
        # Merge predictions with customer features
        print("🔗 Step 3/3: Merging data and caching...")
        customer_data = customer_features.merge(predictions, on='customer_id', how='left')
        
        # Ensure no duplicate customers in final dataset
        duplicate_count = customer_data['customer_id'].duplicated().sum()
        if duplicate_count > 0:
            customer_data = customer_data.drop_duplicates(subset=['customer_id'], keep='first')
            print(f"✅ Final dataset: {len(customer_data)} unique customers (removed {duplicate_count} duplicates)")
        else:
            print(f"✅ Final dataset: {len(customer_data)} unique customers")
        
        # Cache the result both in memory and on disk
        result = customer_data.to_json(date_format='iso', orient='split')
        CACHED_DATA = result
        CACHE_TIMESTAMP = datetime.now()
        
        # Save to persistent cache for next time
        save_persistent_cache(result)
        
        print("✅ Data loaded and cached successfully")
        return result
        
    except Exception as e:
        print(f"❌ Error loading raw data: {e}")
        return '{}'
    finally:
        load_raw_data._loading = False

# Fast date-dependent processing
@app.callback(
    Output('processed-data-store', 'children'),
    [Input('raw-data-store', 'children'),
     Input('analysis-date', 'date')]
)
def process_data_for_date(raw_data_json, analysis_date):
    """Fast processing that only recalculates date-dependent metrics"""
    if not raw_data_json or raw_data_json == '{}':
        return '{}'
    
    try:
        print(f"⚡ Fast processing for date: {analysis_date}")
        
        # Load the pre-computed data
        customer_data = pd.read_json(raw_data_json, orient='split')
        
        # Use selected analysis date for calculations
        if analysis_date:
            analysis_datetime = datetime.strptime(analysis_date, '%Y-%m-%d')
        else:
            analysis_datetime = datetime.now()
        
        # Ensure last_order_date is datetime
        customer_data['last_order_date'] = pd.to_datetime(customer_data['last_order_date'])
        
        # Recalculate days since last order based on analysis date
        customer_data['days_since_last_order'] = (
            analysis_datetime - customer_data['last_order_date']
        ).dt.days
        
        # Focus on actionable timeframe (30-180 days since last order)
        actionable_customers = customer_data[
            (customer_data['days_since_last_order'] >= 30) & 
            (customer_data['days_since_last_order'] <= 180)
        ].copy()
        
        # Create business-relevant segments
        def categorize_timeframe(days):
            if 30 <= days < 60:
                return 'recent'
            elif 60 <= days < 90:
                return 'watch'
            elif 90 <= days <= 180:
                return 'immediate'
            else:
                return 'other'
        
        actionable_customers['timeframe_category'] = actionable_customers['days_since_last_order'].apply(categorize_timeframe)
        
        print("✅ Fast processing complete")
        return actionable_customers.to_json(date_format='iso', orient='split')
        
    except Exception as e:
        print(f"❌ Error processing data for date: {e}")
        return '{}'

# Executive metrics callback
@app.callback(
    [Output('immediate-risk', 'children'),
     Output('watch-list', 'children'),
     Output('revenue-opportunity', 'children'),
     Output('repeat-rate', 'children')],
    Input('processed-data-store', 'children')
)
def update_business_metrics(data_json):
    """Update business-focused metrics"""
    if not data_json or data_json == '{}':
        return '0', '0', '$0', '0%'
    
    try:
        data = pd.read_json(data_json, orient='split')
        
        # Immediate risk customers (90-180 days)
        immediate_risk = len(data[data['timeframe_category'] == 'immediate'])
        
        # Watch list customers (60-90 days)
        watch_list = len(data[data['timeframe_category'] == 'watch'])
        
        # Revenue opportunity (high-value at-risk customers)
        revenue_opportunity = data[
            data['timeframe_category'].isin(['immediate', 'watch'])
        ]['total_spent'].sum()
        
        # Calculate repeat rate (customers with more than 1 order)
        repeat_customers = len(data[data['total_orders'] > 1])
        total_customers = len(data)
        repeat_rate = (repeat_customers / total_customers) if total_customers > 0 else 0
        
        return f"{immediate_risk:,}", f"{watch_list:,}", f"${revenue_opportunity:,.0f}", f"{repeat_rate:.0%}"
        
    except Exception as e:
        print(f"Error updating business metrics: {e}")
        return '0', '0', '$0', '0%'

# Customer table callback
@app.callback(
    [Output('customer-table', 'data'),
     Output('state-filter', 'options')],
    [Input('processed-data-store', 'children'),
     Input('timeframe-filter', 'value'),
     Input('state-filter', 'value')]
)
def update_customer_table(data_json, timeframe_filter, state_filter):
    """Update customer table with business-focused filtering"""
    if not data_json or data_json == '{}':
        return [], []
    
    try:
        data = pd.read_json(data_json, orient='split')
        
        # Apply timeframe filter
        if timeframe_filter == 'immediate':
            filtered_data = data[data['timeframe_category'] == 'immediate']
        elif timeframe_filter == 'watch':
            filtered_data = data[data['timeframe_category'] == 'watch']
        elif timeframe_filter == 'recent':
            filtered_data = data[data['timeframe_category'] == 'recent']
        else:  # all_active
            filtered_data = data
        
        # Filter out non-US states (keep only standard US state codes)
        us_states = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
        }
        
        # Filter data to US states only
        filtered_data = filtered_data[filtered_data['shipping_state'].isin(us_states)]
        
        # Apply state filter
        if state_filter:
            filtered_data = filtered_data[filtered_data['shipping_state'] == state_filter]
        
        # Prepare table data and ensure unique customers
        table_data = filtered_data[[
            'customer_id', 'shipping_state', 'days_since_last_order', 'churn_probability', 
            'jason_risk_score', 'total_spent', 'total_orders', 'product_form', 'total_refunded'
        ]].copy()
        
        # Check for duplicates before processing
        duplicate_count = table_data['customer_id'].duplicated().sum()
        if duplicate_count > 0:
            # Keep the first occurrence of each customer (should be the same data anyway)
            table_data = table_data.drop_duplicates(subset=['customer_id'], keep='first')
        
        # Fill missing refund data and ensure proper data types for filtering
        table_data['total_refunded'] = table_data['total_refunded'].fillna(0)
        
        # Convert ML probability to percentage for display
        table_data['churn_probability_percent'] = (table_data['churn_probability'] * 100).round(0)
        
        # Ensure proper data types for table filtering
        table_data['customer_id'] = table_data['customer_id'].astype(str)
        table_data['days_since_last_order'] = pd.to_numeric(table_data['days_since_last_order'], errors='coerce').fillna(0)
        table_data['churn_probability_percent'] = pd.to_numeric(table_data['churn_probability_percent'], errors='coerce').fillna(0)
        table_data['jason_risk_score'] = pd.to_numeric(table_data['jason_risk_score'], errors='coerce').fillna(0)
        table_data['total_spent'] = pd.to_numeric(table_data['total_spent'], errors='coerce').fillna(0)
        table_data['total_refunded'] = pd.to_numeric(table_data['total_refunded'], errors='coerce').fillna(0)
        table_data['total_orders'] = pd.to_numeric(table_data['total_orders'], errors='coerce').fillna(0)
        
        table_data = table_data.rename(columns={'shipping_state': 'state'})
        
        
        # Sort by risk (ML prediction + days since order)
        table_data = table_data.sort_values(['churn_probability', 'days_since_last_order'], ascending=[False, False])
        
        # State options - only US states
        valid_states = [state for state in sorted(data['shipping_state'].dropna().unique()) if state in us_states]
        state_options = [{'label': state, 'value': state} for state in valid_states]
        
        return table_data.to_dict('records'), state_options
        
    except Exception as e:
        print(f"Error updating customer table: {e}")
        return [], []

# Risk indicators and section description callback
@app.callback(
    [Output('risk-indicators', 'children'),
     Output('risk-analysis-description', 'children')],
    [Input('processed-data-store', 'children'),
     Input('analysis-date', 'date')]
)
def update_risk_indicators(data_json, analysis_date):
    """Update key risk indicators with business context"""
    if not data_json or data_json == '{}':
        section_description = "Data-driven insights into the factors that predict when cart customers won't make a second purchase"
        return "No data available", section_description
    
    try:
        data = pd.read_json(data_json, orient='split')
        at_risk = data[data['timeframe_category'].isin(['immediate', 'watch'])]
        
        indicators = []
        
        # Single order vulnerability - key pattern
        single_order = (at_risk['total_orders'] == 1).mean() * 100
        indicators.append(f"• {single_order:.0f}% are first-time buyers who never returned (high vulnerability)")
        
        # Purchase timing pattern
        avg_days = at_risk['days_since_last_order'].mean()
        if avg_days > 120:
            indicators.append(f"• Average {avg_days:.0f} days since last order (beyond typical repeat cycle)")
        else:
            indicators.append(f"• Average {avg_days:.0f} days since last order (still in potential return window)")
        
        # Value segment analysis
        median_spent = at_risk['total_spent'].median()
        low_value = (at_risk['total_spent'] < median_spent).mean() * 100
        if low_value > 60:
            indicators.append(f"• {low_value:.0f}% spent below median ${median_spent:.0f} (price-sensitive segment)")
        
        # Support interaction patterns (always show if available)
        high_behavior_score = (at_risk['jason_risk_score'] >= 2).mean() * 100
        indicators.append(f"• {high_behavior_score:.0f}% have elevated behavior risk scores (support tickets/issues)")
        
        # Refund/return patterns (always show if available)
        if 'total_refunded' in at_risk.columns:
            has_refunds = (at_risk['total_refunded'] > 0).mean() * 100
            indicators.append(f"• {has_refunds:.0f}% have previous refunds (unmet expectations)")
        
        # Product preference patterns (always show top product)
        if 'product_form' in at_risk.columns:
            product_dist = at_risk['product_form'].value_counts(normalize=True)
            if len(product_dist) > 0:
                top_product = product_dist.index[0]
                top_pct = product_dist.iloc[0] * 100
                indicators.append(f"• {top_pct:.0f}% purchased {top_product} (most common product)")
        
        # High-risk customers by ML score (always show)
        high_ml_risk = (at_risk['churn_probability'] >= 0.7).mean() * 100
        indicators.append(f"• {high_ml_risk:.0f}% have >70% ML churn probability (very high risk)")
        
        # Average order value pattern (always show)
        if len(at_risk) > 0:
            avg_order_value = at_risk['total_spent'] / at_risk['total_orders'].replace(0, 1)  # Avoid division by zero
            low_aov = (avg_order_value < avg_order_value.median()).mean() * 100
            indicators.append(f"• {low_aov:.0f}% have below-median order values (price sensitivity)")
        
        # Acquisition channel insights (always show top channel)
        if 'primary_utm_source' in at_risk.columns:
            utm_dist = at_risk['primary_utm_source'].value_counts(normalize=True)
            if len(utm_dist) > 0:
                top_utm = utm_dist.index[0]
                top_utm_pct = utm_dist.iloc[0] * 100
                indicators.append(f"• {top_utm_pct:.0f}% from {top_utm} channel (top acquisition source)")
        
        indicators_div = html.Div([html.P(indicator, style={'margin': '0.5rem 0', 'fontSize': '14px'}) 
                        for indicator in indicators[:7]])
        
        # Create section description with date context
        date_str = datetime.strptime(analysis_date, '%Y-%m-%d').strftime('%B %Y') if analysis_date else datetime.now().strftime('%B %Y')
        section_description = f"Data-driven insights into factors that predict when cart customers won't make a second purchase. Analysis based on customers active within 30-180 days of {date_str}."
        
        return indicators_div, section_description
        
    except Exception as e:
        print(f"Error updating risk indicators: {e}")
        section_description = "Data-driven insights into the factors that predict when cart customers won't make a second purchase"
        return "Error loading risk indicators", section_description

# Feature importance chart (fixed column names)
@app.callback(
    Output('feature-importance-chart', 'figure'),
    Input('processed-data-store', 'children')
)
def update_feature_importance(data_json):
    """Update feature importance chart with correct column names"""
    if not data_json or data_json == '{}':
        return {}
    
    try:
        # Get feature importance from model
        feature_impact = churn_predictor.get_feature_impact_summary()
        
        
        if not feature_impact.empty and 'importance' in feature_impact.columns:
            # Clean and validate data
            top_features = feature_impact.head(8).copy()
            
            # Ensure no null values
            top_features = top_features.dropna()
            
            # Ensure feature names are strings and importance values are numeric
            if 'feature' in top_features.columns:
                top_features['feature'] = top_features['feature'].astype(str)
            
            top_features['importance'] = pd.to_numeric(top_features['importance'], errors='coerce')
            top_features = top_features.dropna()
            
            # Ensure all values are finite
            top_features = top_features[np.isfinite(top_features['importance'])]
            
            if len(top_features) > 0:
                # Create a simple DataFrame to ensure clean data
                clean_data = pd.DataFrame({
                    'feature': top_features['feature'].tolist(),
                    'importance': top_features['importance'].tolist()
                })
                
                # Use go.Bar instead of px.bar to avoid template issues
                fig = go.Figure(data=[
                    go.Bar(
                        x=clean_data['importance'],
                        y=clean_data['feature'],
                        orientation='h',
                        marker_color='#3b82f6'
                    )
                ])
                
                fig.update_xaxes(title_text='Predictive Importance')
                fig.update_yaxes(title_text='Customer Factors')
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis={'categoryorder': 'total ascending'},
                    font_size=12
                )
                
                return fig
            else:
                # No valid data after cleaning
                fig = go.Figure()
                fig.add_annotation(
                    text="No valid feature importance data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font_size=14
                )
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                return fig
        else:
            # Create placeholder chart
            fig = go.Figure()
            fig.add_annotation(
                text="Feature importance analysis not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=14
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            return fig
        
    except Exception as e:
        return {}

# Behavior score chart
@app.callback(
    Output('behavior-score-chart', 'figure'),
    Input('processed-data-store', 'children')
)
def update_behavior_score_chart(data_json):
    """Update behavior score distribution"""
    if not data_json or data_json == '{}':
        return {}
    
    try:
        data = pd.read_json(data_json, orient='split')
        
        # Check if jason_risk_score column exists and has data
        if 'jason_risk_score' not in data.columns or data['jason_risk_score'].isna().all():
            fig = go.Figure()
            fig.add_annotation(
                text="Behavior score data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=14
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            return fig
        
        # Analyze behavior score distribution
        score_dist = data['jason_risk_score'].value_counts().sort_index()
        
        if score_dist.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No behavior score data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=14
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            return fig
        
        # Convert to proper format for plotly - ensure clean data
        score_data = pd.DataFrame({
            'score': [str(x) for x in score_dist.index],
            'count': [int(x) for x in score_dist.values]
        })
        
        # Remove any invalid entries
        score_data = score_data.dropna()
        
        if len(score_data) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No behavior score data to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=14
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            return fig
        
        # Use go.Bar instead of px.bar to avoid template issues
        fig = go.Figure(data=[
            go.Bar(
                x=score_data['score'],
                y=score_data['count'],
                marker_color='#6366f1'
            )
        ])
        
        fig.update_xaxes(title_text='Behavior Risk Score')
        fig.update_yaxes(title_text='Number of Customers')
        
        # Add vertical line at score 2 for elevated risk threshold
        if '2' in score_data['score'].values:
            score_2_position = list(score_data['score']).index('2')
            fig.add_vline(
                x=score_2_position, 
                line_dash="dash", 
                line_color="orange",
                annotation_text="Elevated Risk",
                annotation_position="top"
            )
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            font_size=12
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating behavior score chart: {e}")
        return {}

# Geographic success patterns
@app.callback(
    Output('geographic-success-chart', 'figure'),
    [Input('processed-data-store', 'children'),
     Input('analysis-date', 'date')]
)
def update_geographic_success_chart(data_json, analysis_date):
    """Show which states have best/worst retention"""
    if not data_json or data_json == '{}':
        return {}
    
    try:
        data = pd.read_json(data_json, orient='split')
        
        # Filter to US states only
        us_states = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
        }
        
        us_data = data[data['shipping_state'].isin(us_states)]
        
        # Calculate repeat purchase rate by state
        if len(us_data) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No US customer data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=14
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            return fig
        
        state_success = us_data.groupby('shipping_state').agg({
            'total_orders': lambda x: (x >= 2).mean(),  # Repeat rate (2+ orders)
            'customer_id': 'count'
        }).reset_index()
        
        state_success.columns = ['state', 'repeat_rate', 'customer_count']
        state_success = state_success[state_success['customer_count'] >= 5]  # Show more states
        
        # Clean data - remove any invalid values
        state_success = state_success.dropna()
        state_success['repeat_rate'] = pd.to_numeric(state_success['repeat_rate'], errors='coerce')
        state_success = state_success.dropna()
        
        if len(state_success) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient state data for analysis (need 5+ customers per state)",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=12
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            return fig
        
        state_success = state_success.sort_values('repeat_rate', ascending=False)  # Show all states
        
        
        # Ensure clean data types for plotting
        clean_geographic_data = pd.DataFrame({
            'state': state_success['state'].astype(str).tolist(),
            'repeat_rate': state_success['repeat_rate'].astype(float).tolist()
        })
        
        # Use go.Bar instead of px.bar to avoid template issues
        fig = go.Figure(data=[
            go.Bar(
                x=clean_geographic_data['state'],
                y=clean_geographic_data['repeat_rate'],
                marker_color='#10b981'
            )
        ])
        
        fig.update_xaxes(title_text='State')
        fig.update_yaxes(title_text='Repeat Purchase Rate')
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=40),
            font_size=12
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating geographic success chart: {e}")
        return {}

# Journey segments (UMAP with business context)
@app.callback(
    Output('journey-segments-chart', 'figure'),
    Input('processed-data-store', 'children')
)
def update_journey_segments_chart(data_json):
    """Show customer segments with business interpretation"""
    if not data_json or data_json == '{}':
        return {}
    
    try:
        data = pd.read_json(data_json, orient='split')
        
        if 'umap_1' in data.columns and 'umap_2' in data.columns:
            # Analyze cluster characteristics for meaningful labels
            cluster_analysis = {}
            for cluster in data['umap_cluster'].unique():
                cluster_data = data[data['umap_cluster'] == cluster]
                cluster_analysis[cluster] = {
                    'avg_spent': cluster_data['total_spent'].mean(),
                    'avg_orders': cluster_data['total_orders'].mean(),
                    'avg_days_since': cluster_data['days_since_last_order'].mean(),
                    'repeat_rate': (cluster_data['total_orders'] > 1).mean(),
                    'avg_behavior_score': cluster_data['jason_risk_score'].mean()
                }
            
            # Create descriptive labels based on characteristics - ensure unique labels
            cluster_labels = {}
            used_labels = set()
            available_labels = [
                'High-Value Loyalists', 'High-Risk Customers', 'Recent Active', 
                'Moderate Loyalists', 'Low-Spend Customers', 'Standard Customers',
                'Price-Sensitive', 'Occasional Buyers', 'New Customers'
            ]
            
            # Sort clusters by characteristics for consistent labeling
            sorted_clusters = sorted(cluster_analysis.items(), 
                                   key=lambda x: (x[1]['repeat_rate'], x[1]['avg_spent']), 
                                   reverse=True)
            
            for cluster, stats in sorted_clusters:
                if stats['repeat_rate'] > 0.3 and stats['avg_spent'] > data['total_spent'].median() and 'High-Value Loyalists' not in used_labels:
                    label = 'High-Value Loyalists'
                elif stats['avg_behavior_score'] > 1.5 and 'High-Risk Customers' not in used_labels:
                    label = 'High-Risk Customers'
                elif stats['avg_days_since'] < 90 and 'Recent Active' not in used_labels:
                    label = 'Recent Active'
                elif stats['repeat_rate'] > 0.2 and 'Moderate Loyalists' not in used_labels:
                    label = 'Moderate Loyalists'
                elif stats['avg_spent'] < data['total_spent'].quantile(0.25) and 'Low-Spend Customers' not in used_labels:
                    label = 'Low-Spend Customers'
                else:
                    # Assign first available label for remaining clusters
                    label = next((l for l in available_labels if l not in used_labels), f'Segment {cluster}')
                
                cluster_labels[cluster] = label
                used_labels.add(label)
            
            data['segment_label'] = data['umap_cluster'].map(cluster_labels)
            
            fig = px.scatter(
                data.sample(n=min(1000, len(data))),
                x='umap_1',
                y='umap_2',
                color='segment_label',
                size='total_spent',
                hover_data=['days_since_last_order', 'total_orders', 'jason_risk_score'],
                labels={'umap_1': 'Customer Behavior Pattern 1', 'umap_2': 'Customer Behavior Pattern 2'}
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                font_size=12
            )
            
            return fig
        else:
            # Create placeholder
            fig = go.Figure()
            fig.add_annotation(
                text="Customer segmentation analysis not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=14
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            return fig
        
    except Exception as e:
        print(f"Error updating journey segments chart: {e}")
        return {}

# Product success analysis
@app.callback(
    Output('product-success-chart', 'figure'),
    [Input('processed-data-store', 'children'),
     Input('analysis-date', 'date')]
)
def update_product_success_chart(data_json, analysis_date):
    """Show which products have best retention"""
    if not data_json or data_json == '{}':
        return {}
    
    try:
        data = pd.read_json(data_json, orient='split')
        
        if 'product_form' in data.columns and len(data) > 0:
            # Remove any null/empty product forms
            data_clean = data.dropna(subset=['product_form'])
            data_clean = data_clean[data_clean['product_form'].str.strip() != '']
            
            if len(data_clean) == 0:
                fig = go.Figure()
                fig.add_annotation(
                    text="No valid product form data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font_size=14
                )
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                return fig
            
            product_success = data_clean.groupby('product_form').agg({
                'total_orders': lambda x: (x > 1).mean(),  # Repeat rate
                'customer_id': 'count'
            }).reset_index()
            
            product_success.columns = ['product', 'repeat_rate', 'customer_count']
            product_success = product_success[product_success['customer_count'] >= 10]  # Lower threshold to show more categories
            product_success = product_success.sort_values('repeat_rate', ascending=False)  # Sort descending
            
            if len(product_success) > 0:
                
                # Ensure clean data types for plotting
                clean_product_data = pd.DataFrame({
                    'product': product_success['product'].astype(str).tolist(),
                    'repeat_rate': product_success['repeat_rate'].astype(float).tolist()
                })
                
                # Use go.Bar instead of px.bar to avoid template issues
                fig = go.Figure(data=[
                    go.Bar(
                        x=clean_product_data['product'],
                        y=clean_product_data['repeat_rate'],
                        marker_color='#f59e0b'
                    )
                ])
                
                fig.update_xaxes(title_text='Product Type')
                fig.update_yaxes(title_text='Repeat Purchase Rate')
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=40),
                    font_size=12
                )
                
                return fig
            else:
                # Not enough data for analysis
                fig = go.Figure()
                fig.add_annotation(
                    text="Insufficient data for product analysis (need 10+ customers per product type)",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font_size=12
                )
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                return fig
        else:
            # Create placeholder
            fig = go.Figure()
            fig.add_annotation(
                text="Product analysis not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=14
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            return fig
        
    except Exception as e:
        print(f"Error updating product success chart: {e}")
        return {}

# Priority actions
@app.callback(
    Output('priority-actions', 'children'),
    Input('processed-data-store', 'children')
)
def update_priority_actions(data_json):
    """Generate actionable business recommendations"""
    if not data_json or data_json == '{}':
        return "No data available"
    
    try:
        data = pd.read_json(data_json, orient='split')
        
        actions = []
        
        # Immediate contact list
        immediate_risk = data[data['timeframe_category'] == 'immediate']
        if len(immediate_risk) > 0:
            high_value_immediate = immediate_risk[
                immediate_risk['total_spent'] > immediate_risk['total_spent'].quantile(0.7)
            ]
            actions.append(f"• Contact {len(high_value_immediate)} high-value customers (90-180 days) with personalized offers")
        
        # Product issue follow-up
        if 'product_issue' in data.columns:
            product_issues = data[
                (data['product_issue'] == 1) & 
                (data['timeframe_category'].isin(['immediate', 'watch']))
            ]
            if len(product_issues) > 0:
                actions.append(f"• Follow up with {len(product_issues)} customers who had product issues")
        
        # Behavior score intervention
        high_behavior_risk = data[
            (data['jason_risk_score'] >= 2) & 
            (data['timeframe_category'].isin(['immediate', 'watch']))
        ]
        if len(high_behavior_risk) > 0:
            actions.append(f"• Special attention needed for {len(high_behavior_risk)} customers with elevated behavior risk")
        
        return html.Div([html.P(action, style={'margin': '0.5rem 0', 'fontSize': '14px'}) 
                        for action in actions[:4]])
        
    except Exception as e:
        print(f"Error updating priority actions: {e}")
        return "Error loading priority actions"

# Segment strategies
@app.callback(
    Output('segment-strategies', 'children'),
    Input('processed-data-store', 'children')
)
def update_segment_strategies(data_json):
    """Provide segment-specific strategies"""
    if not data_json or data_json == '{}':
        return "No data available"
    
    try:
        data = pd.read_json(data_json, orient='split')
        
        strategies = []
        
        if 'umap_cluster' in data.columns:
            for cluster in data['umap_cluster'].unique():
                cluster_data = data[data['umap_cluster'] == cluster]
                avg_spent = cluster_data['total_spent'].mean()
                repeat_rate = (cluster_data['total_orders'] > 1).mean()
                
                # Provide more actionable insights
                risk_level = "High" if cluster_data['jason_risk_score'].mean() >= 2 else "Medium" if cluster_data['jason_risk_score'].mean() >= 1 else "Low"
                recency = "Recent" if cluster_data['days_since_last_order'].mean() < 90 else "Distant"
                
                strategies.append(
                    f"• Segment {cluster}: {len(cluster_data)} customers, "
                    f"${avg_spent:.0f} avg spent, {repeat_rate:.0%} repeat rate, {risk_level} risk, {recency} activity"
                )
        else:
            strategies.append("• Segment analysis not available - consider implementing customer clustering")
        
        return html.Div([html.P(strategy, style={'margin': '0.5rem 0', 'fontSize': '14px'}) 
                        for strategy in strategies])
        
    except Exception as e:
        print(f"Error updating segment strategies: {e}")
        return "Error loading segment strategies"

# Success patterns
@app.callback(
    Output('success-patterns', 'children'),
    Input('processed-data-store', 'children')
)
def update_success_patterns(data_json):
    """Identify what makes customers successful"""
    if not data_json or data_json == '{}':
        return "No data available"
    
    try:
        data = pd.read_json(data_json, orient='split')
        
        patterns = []
        
        # More meaningful purchase pattern analysis
        repeat_customers = data[data['total_orders'] > 1]
        single_customers = data[data['total_orders'] == 1]
        
        if len(repeat_customers) > 0 and len(single_customers) > 0:
            # Focus on more significant differences
            repeat_total_spent = repeat_customers['total_spent'].mean()
            single_total_spent = single_customers['total_spent'].mean()
            
            # Only show if there's a meaningful difference (>10%)
            if abs(repeat_total_spent - single_total_spent) / single_total_spent > 0.1:
                patterns.append(f"• Repeat customers spend ${repeat_total_spent:.0f} on average vs ${single_total_spent:.0f} for one-time buyers")
            
            # Analyze time patterns
            repeat_avg_days = repeat_customers['days_since_last_order'].mean()
            single_avg_days = single_customers['days_since_last_order'].mean()
            
            if abs(repeat_avg_days - single_avg_days) > 10:
                patterns.append(f"• Repeat customers last ordered {repeat_avg_days:.0f} days ago vs {single_avg_days:.0f} days for one-time buyers")
        
        # Product form success patterns
        if 'product_form' in data.columns and len(data) > 0:
            product_repeat_rates = data.groupby('product_form')['total_orders'].apply(lambda x: (x > 1).mean())
            if not product_repeat_rates.empty:
                best_product = product_repeat_rates.idxmax()
                best_rate = product_repeat_rates.max()
                patterns.append(f"• {best_product} has highest repeat rate at {best_rate:.0%}")
        
        # Geographic success - US states only with minimum customer threshold
        us_states = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
        }
        us_data = data[data['shipping_state'].isin(us_states)]
        
        # Apply same minimum customer filter as the chart (5+ customers)
        state_success = us_data.groupby('shipping_state').agg({
            'total_orders': lambda x: (x > 1).mean(),  # Repeat rate
            'customer_id': 'count'
        }).reset_index()
        state_success.columns = ['state', 'repeat_rate', 'customer_count']
        state_success_filtered = state_success[state_success['customer_count'] >= 5]
        
        if len(state_success_filtered) > 0:
            best_state_row = state_success_filtered.loc[state_success_filtered['repeat_rate'].idxmax()]
            best_state = best_state_row['state']
            # Convert state code to full name for display
            state_names = {
                'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
                'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
                'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
                'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
                'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
                'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
                'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
                'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
                'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
                'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'Washington DC'
            }
            best_state_name = state_names.get(best_state, best_state)
            patterns.append(f"• {best_state_name} customers have highest repeat purchase success")
        
        return html.Div([html.P(pattern, style={'margin': '0.5rem 0', 'fontSize': '14px'}) 
                        for pattern in patterns[:4]])
        
    except Exception as e:
        print(f"Error updating success patterns: {e}")
        return "Error loading success patterns"

# Export functionality - John's approach for top 10% high-risk customers
@app.callback(
    [Output('export-status', 'children'),
     Output('export-btn', 'children')],
    Input('export-btn', 'n_clicks'),
    Input('processed-data-store', 'children'),
    prevent_initial_call=True
)
def export_high_risk_customers(n_clicks, data_json):
    """Export top 10% high-risk customers with download link"""
    if not n_clicks or not data_json or data_json == '{}':
        return "", "📤 Export Top 10% High-Risk"
    
    try:
        data = pd.read_json(data_json, orient='split')
        
        # Calculate top 10% cutoff like John's approach
        cutoff = data['churn_probability'].quantile(0.90)
        top_10_risk = data[data['churn_probability'] >= cutoff].copy()
        top_10_risk = top_10_risk.sort_values('churn_probability', ascending=False)
        
        # Select relevant columns for export (similar to John's selection)
        cols_to_export = [
            'customer_id', 'churn_probability', 'jason_risk_score', 
            'days_since_last_order', 'total_spent', 'total_refunded', 
            'total_orders', 'product_form', 'shipping_state'
        ]
        
        # Add John's features if available
        john_cols = ['ever_used_discount', 'discount_refund_interaction', 'site_flag', 'escalated_flag']
        for col in john_cols:
            if col in top_10_risk.columns:
                cols_to_export.append(col)
        
        # Filter to available columns
        available_cols = [col for col in cols_to_export if col in top_10_risk.columns]
        
        # Create CSV content for download
        csv_content = top_10_risk[available_cols].to_csv(index=False)
        filename = f"high_risk_customers_top10pct_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        
        # Create download link
        download_link = html.A(
            f"💾 Download {filename} ({len(top_10_risk)} customers)",
            href=f"data:text/csv;charset=utf-8,{csv_content}",
            download=filename,
            style={'color': '#059669', 'textDecoration': 'underline', 'fontSize': '0.9rem'}
        )
        
        status_msg = html.Div([
            html.P(f"✅ Ready for download:", style={'margin': '0', 'fontSize': '0.8rem'}),
            download_link
        ])
        
        return status_msg, "📤 Export Again"
        
    except Exception as e:
        print(f"Error exporting customers: {e}")
        return f"❌ Export failed: {str(e)}", "📤 Export Top 10% High-Risk"

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)