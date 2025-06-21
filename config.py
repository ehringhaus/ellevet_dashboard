"""
Configuration settings for ElleVet Customer Churn Dashboard
"""

import os
from datetime import datetime

# Dashboard Configuration
CONFIG = {
    # Data Settings
    'data_path': 'data/',
    'model_path': 'models/',
    
    # Refresh Settings
    'refresh_interval': 3600,  # 1 hour in seconds
    'auto_retrain_days': 7,    # Retrain model every 7 days
    
    # Business Rules - Updated to match current dashboard
    'churn_threshold_days': 45,  # Customer is churned if no order in 45 days (adaptive)
    'immediate_risk_days': [90, 180],  # Immediate attention timeframe
    'watch_list_days': [60, 90],       # Watch list timeframe
    'recent_activity_days': [30, 60],  # Recent activity timeframe
    'actionable_window_days': [30, 180], # Overall actionable window
    'high_risk_threshold': 0.7,  # Probability threshold for high risk
    'medium_risk_threshold': 0.4, # Probability threshold for medium risk
    
    # Dashboard Display - Updated to current settings
    'max_table_rows': 12,      # Maximum rows to display in risk table (paginated)
    'show_all_states': True,   # Show all US states in geographic chart
    'min_customers_per_state': 5,  # Minimum customers to show state in chart
    'min_customers_per_product': 10, # Minimum customers to show product in chart
    
    # Model Settings
    'test_size': 0.2,          # Train/test split ratio
    'random_state': 42,        # For reproducible results
    'cv_folds': 5,             # Cross-validation folds
    
    # Alert Thresholds
    'high_risk_alert_count': 50,  # Alert if more than 50 high-risk customers
    'churn_rate_alert': 0.6,      # Alert if churn rate exceeds 60%
    
    # Export Settings
    'export_filename': f'at_risk_customers_{datetime.now().strftime("%Y%m%d")}.csv',
    
    # Color Scheme
    'colors': {
        'primary': '#1f2937',      # Dark blue-gray
        'secondary': '#6b7280',    # Medium gray
        'success': '#10b981',      # Green
        'warning': '#f59e0b',      # Orange
        'danger': '#ef4444',       # Red
        'info': '#3b82f6',         # Blue
        'light': '#f9fafb',        # Light gray
        'white': '#ffffff'
    },
    
    # Chart Settings - Updated to match dashboard
    'chart_height': 300,
    'chart_template': 'plotly_white',
    'chart_responsive': True,
    'chart_margin': {'l': 20, 'r': 20, 't': 20, 'b': 20},
    
    # Caching Settings
    'cache_duration_minutes': 5,  # Cache heavy computations for 5 minutes
    
    # US States Filter
    'us_states': {
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
    },
    
    # Email Settings (for future notifications)
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'notification_emails': [
        'analytics@ellevetsciences.com',
        'marketing@ellevetsciences.com'
    ]
}

# Feature Configuration
FEATURE_CONFIG = {
    'required_features': [
        'total_orders', 'total_spent', 'days_since_first_order', 'days_since_last_order',
        'customer_lifetime_days', 'avg_order_value', 'shipping_state'
    ],
    
    'optional_features': [
        'order_value_std', 'min_order_value', 'max_order_value', 'avg_shipping',
        'total_discounts', 'total_tax', 'primary_channel', 'primary_payment',
        'primary_utm_source', 'primary_utm_medium', 'refund_count', 'total_refunded',
        'ticket_count'
    ],
    
    'engineered_features': [
        'avg_days_between_orders', 'order_frequency', 'spend_per_day',
        'has_refunds', 'refund_rate'
    ]
}

# Database Configuration (if using database instead of CSV)
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', 5432),
    'database': os.getenv('DB_NAME', 'elleevet'),
    'username': os.getenv('DB_USER', 'analytics'),
    'password': os.getenv('DB_PASSWORD', ''),
    'connection_timeout': 30
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'logs/dashboard.log',
    'max_file_size': 10485760,  # 10MB
    'backup_count': 5
}

# Security Configuration
SECURITY_CONFIG = {
    'require_authentication': False,  # Set to True for production
    'session_timeout': 3600,          # 1 hour
    'allowed_ip_ranges': ['0.0.0.0/0'],  # Allow all IPs (restrict in production)
    'api_rate_limit': 100             # Requests per minute
}

# Notification Templates
NOTIFICATION_TEMPLATES = {
    'high_risk_alert': {
        'subject': 'ElleVet Alert: High Number of At-Risk Customers',
        'body': '''
        Alert: {count} customers are currently at high risk of churning.
        
        Total revenue at risk: ${revenue:,.0f}
        Current churn rate: {churn_rate:.1%}
        
        Recommended actions:
        - Review high-risk customer list in dashboard
        - Launch retention campaign immediately
        - Consider personalized discount offers
        
        Dashboard link: {dashboard_url}
        '''
    },
    
    'model_performance_alert': {
        'subject': 'ElleVet Alert: Model Performance Degradation',
        'body': '''
        Alert: Churn prediction model performance has degraded.
        
        Current accuracy: {accuracy:.1%}
        Recommended threshold: 80%+
        
        Action required: Model retraining recommended
        
        Dashboard link: {dashboard_url}
        '''
    }
}

# Export column mapping for customer lists
# Export column mapping - Updated to match current dashboard
EXPORT_COLUMNS = {
    'customer_id': 'Customer ID',
    'shipping_state': 'State',
    'churn_probability': 'ML Risk %',
    'jason_risk_score': 'Behavior Score',
    'days_since_last_order': 'Days Since Last Order',
    'total_spent': 'Total Spent $',
    'total_refunded': 'Total Refunded $',
    'total_orders': 'Orders',
    'product_form': 'Product Type',
    'avg_order_value': 'Avg Order Value',
    'primary_utm_source': 'Acquisition Source',
    'timeframe_category': 'Risk Timeframe'
}

# State abbreviation mapping - Updated to include DC
STATE_MAPPING = {
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

# Jason's Risk Score Configuration
JASON_RISK_CONFIG = {
    'cancel_tag_points': 2,      # Strong churn signal
    'product_issue_points': 1,   # Product dissatisfaction  
    'urgent_tag_points': 1,      # Immediate concern
    'multiple_tickets_points': 1, # Ongoing issues (2+ tickets)
    'refund_points': 1,          # Unmet expectations
    'utm_risk_points': 1,        # Price-sensitive acquisition
    'low_spend_threshold': 40,   # Below this is +1 point
    'long_gap_threshold': 120,   # Days since order is +1 point
    'max_score': 8              # Theoretical maximum
}