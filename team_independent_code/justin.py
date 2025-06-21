# %%
# customers_redacted.csv
################################ customers_df ################################
# site: String - Website/platform where the customer registered
# customer_key: String - Unique identifier for customer in the system
# customer_id: Integer - Numeric identifier for the customer
# shipping_city: String - Customer's shipping city
# shipping_state: String - Customer's shipping state (e.g., NY, CA, FL)
# shipping_postcode: Integer - Customer's shipping postal code
# shipping_country: String - Customer's shipping country
# roles_json: String - JSON containing customer roles or status information
# primary_role: String - Primary role of the customer
# raw_capabilities: String - Permissions/capabilities associated with customer
# email_hash: String - Hashed email address for privacy

# customer_shipping_addresses_redacted.csv
################################ customers_df ################################
# site: String - Website/platform where the customer registered
# customer_id: Integer - Numeric identifier for the customer
# shipping_address_hash: String - Hashed shipping address for privacy (useful for identifying unique households/duplicate customers)
#  email_hash: String - Hashed email address matching customers_df

# orders_redacted.csv
################################ orders_df ################################
# order_id: Integer - Unique identifier for the order
# customer_id: Integer - Identifier linking to customer_df
# order_status: String - Current status of the order (e.g., completed, processing)
# order_created_ts: Date - Timestamp when order was created
# order_created_gmt_ts: Date - GMT timestamp when order was created
# order_modified_ts: Date - Timestamp when order was last modified
# order_modified_gmt_ts: Date - GMT timestamp when order was last modified
# site: String - Website/platform where order was placed
# order_key: String - Alternative identifier for the order
# transaction_id: String - Payment transaction identifier
# created_via: String - Channel through which order was created
# completed_date: Date - Date when order was completed
# order_total: Float - Total amount of the order
# order_shipping: Float - Shipping cost
# cart_discount: Float - Discount amount applied to cart
# order_tax: Float - Tax amount
# cart_discount_tax: Integer - Tax on discounted amount
# order_shipping_tax: Float - Tax on shipping
# order_currency: String - Currency of the order
# customer_user_agent: String - Browser/device information
# wwp_wholesale_role: String - Wholesale role if applicable
# wwpp_order_type: String - Order type for wholesale
# wwpp_wholesale_order_type: String - Detailed wholesale order type
# wc_acof_2: String - Additional order field
# billing_city: String - Billing address city
# billing_state: String - Billing address state
# billing_postcode: Integer - Billing address postal code
# billing_country: String - Billing address country
# shipping_city: String - Shipping address city
# shipping_state: String - Shipping address state
# shipping_postcode: Integer - Shipping address postal code
# shipping_country: String - Shipping address country
# payment_method: String - Method of payment
# payment_method_title: String - Display name of payment method
# wc_square_credit_card_card_expiry_date: String - Card expiry date if paid via Square
# paid_date: Date - Date when payment was received
# vet_city: String - Veterinarian's city
# vet_state: String - Veterinarian's state
# vet_postcode: Integer - Veterinarian's postal code
# vet_country: String - Veterinarian's country
# vet_affiliate: String - Veterinarian affiliate information
# customer_role: String - Role of customer for this order
# line_items: String - Products purchased in this order
# coupons_json: String - Coupon information used for order
# is_subscription_renewal: Boolean - Whether order is a subscription renewal
# is_subscription_start: Boolean - Whether order starts a subscription
# parent_subscription_id: String - ID of parent subscription if renewal
# email_hash: String - Hashed email matching customers_df
# vet_place_id_hash: String - Hashed place ID for veterinarian

# orders_with_utm.csv
################################ orders_utm_df ################################
# site: String - Website/platform where order was placed
# order_id: Integer - Unique identifier matching orders_df
# utm_source: String - Source of the traffic (e.g., google, facebook)
# utm_medium: String - Medium of the traffic (e.g., cpc, email)
# utm_campaign: String - Campaign identifier
# utm_content: String - Content identifier within campaign
# utm_term: String - Search terms or ad targeting
# wc_utm_source: String - WooCommerce tracked source
# wc_utm_medium: String - WooCommerce tracked medium
# wc_utm_campaign: String - WooCommerce tracked campaign
# wc_utm_content: String - WooCommerce tracked content
# wc_utm_term: String - WooCommerce tracked term

# quizzes_redacted.csv
################################ quizzes_df ################################
# id: Integer - Unique identifier for quiz submission
# date: String - Date of quiz submission
# email: String - Email address of pet owner (can be joined with email_hash)
# 1._pet_name: String - Name of the pet
# 2._pet_age: String - Age of the pet
# 3._pet_breed: String - Breed of the pet
# 6._pet_gender: String - Gender of the pet
# 7._pet_weight: Integer - Weight of the pet
# 12._pet_support: String - Type of support needed for pet
# 14._pet_nutrition: String - Pet's nutrition needs
# 16._pet_type: String - Type of pet (e.g., dog, cat)
# 17._medications: String - Current medications
# 18._provider: String - Healthcare provider
# 20._none_provider: Boolean - Whether pet has no healthcare provider
# 21._pet_pregnant_or_nursing: String - Whether pet is pregnant or nursing
# 22._pet_build: String - Physical build of pet
# 23._picky_eater: String - Whether pet is a picky eater
# 24._situational_stress: String - Whether pet experiences situational stress
# 25._medication_list: String - List of medications
# 26._insurance: String - Pet insurance information
# 28._results_1: String - First quiz result/recommendation
# 29._results_2: String - Second quiz result/recommendation
# 30._results_3: String - Third quiz result/recommendation
# web_source: String - Source of web traffic
# search_engine: String - Search engine used if applicable
# page_views: Integer - Number of page views
# past_visits: Integer - Number of past website visits
# country: String - Country of quiz taker
# city: String - City of quiz taker
# state: String - State of quiz taker
# postal_code: Integer - Postal code of quiz taker
# device_type: String - Type of device used
# operating_system: String - Operating system used
# browser_type: String - Browser used
# browser_version: String - Version of browser
# submit_url: String - URL where quiz was submitted
# referring_url: String - URL that referred to quiz
# landing_url: String - First page visited on site

# refunds_affiliated.csv
################################ refunds_df ################################
# refund_id: Integer - Unique identifier for the refund
# order_id: Integer - Order ID that was refunded (joins to orders_df)
# refund_date: Date - Date of refund
# refund_amount: Float - Amount refunded
# refund_reason: String - Free-text reason for refund
# refund_reason_dropdown: String - Categorized reason from dropdown
# site: String - Website/platform where refund was processed

# subscription_cancellation_reasons.csv
################################ sub_cancel_df ################################
# site: String - Website/platform where subscription was managed
# id: Integer - Unique identifier for the cancellation record
# cancellation_reason: String - Reason for cancellation
# suspend_reason: String - Reason for suspension if applicable
# cancellation_reason_other: String - Additional details if "other" selected
# suspend_reason_other: String - Additional suspension details if "other"

# subscriptions_redacted.csv
################################ subscriptions_df ################################
# site: String - Website/platform where subscription was created
# subscription_id: Integer - Unique identifier for subscription
# subscription_status: String - Current status (active, cancelled, etc.)
# subscription_billing_period: String - Billing period (e.g., month)
# subscription_billing_interval: Integer - Interval between billings
# subscription_suspended_count: Integer - Number of times subscription was suspended
# subscription_created_ts: Date - Timestamp of subscription creation
# subscription_last_modified_ts: Date - Timestamp of last modification
# subscription_cancelled_ts: String - Timestamp of cancellation if applicable
# subscription_ended_ts: String - Timestamp when subscription ended
# subscription_next_payment_ts: Date - Next scheduled payment date
# subscription_parent_id: Integer - Parent subscription ID if applicable
# subscription_customer_user_id: Integer - Customer ID (joins to customer_id)
# subscription_order_total: Float - Total amount of subscription
# subscription_coupon_uses: String - Coupon usage information
# subscription_payment_method: String - Payment method for subscription
# subscription_cancellation_reason: String - Reason for cancellation
# email_hash: String - Hashed email matching customers_df

# tickets_redacted.csv
################################ tickets_df ################################
# url: String - URL of the support ticket
# id: Integer - Unique identifier for the ticket
# external_id: String - External system identifier
# created_at: Date - Creation timestamp
# updated_at: Date - Last update timestamp
# generated_timestamp: Integer - System-generated timestamp
# type: String - Type of ticket
# priority: String - Priority level
# status: String - Current status
# recipient: String - Recipient of the ticket
# requester_id: Float - ID of person who made request
# requester_email: String - Email of requester (can be joined with email_hash)
# submitter_id: Float - ID of ticket submitter
# assignee_id: Float - ID of assignee
# organization_id: Float - Organization ID
# group_id: Float - Group ID
# collaborator_ids: Float - IDs of collaborators
# follower_ids: String - IDs of followers
# email_cc_ids: Float - CC'd email IDs
# forum_topic_id: String - Related forum topic
# problem_id: String - Related problem ID
# has_incidents: Boolean - Whether ticket has incidents
# is_public: Boolean - Whether ticket is public
# due_at: String - Due date
# tags: String - Tags associated with ticket
# satisfaction_score: String - Customer satisfaction score
# sharing_agreement_ids: String - IDs of sharing agreements
# custom_status_id: Float - Custom status identifier
# encoded_id: String - Encoded ticket ID
# followup_ids: String - IDs of follow-up tickets
# ticket_form_id: Float - Form ID used for ticket
# brand_id: Float - Brand identifier
# allow_channelback: Boolean - Channelback permission
# allow_attachments: Boolean - Attachment permission
# from_messaging_channel: Boolean - If from messaging channel
# via_channel: String - Channel ticket came through

# %%
# Identifying Subscribers vs. One-Time Customers
# Primary Method: Use is_subscription_start and is_subscription_renewal flags in orders_df
# - Subscriber order: is_subscription_start = True OR is_subscription_renewal = True
# - One-time customer order: Both flags = False
# Note: created_via = 'checkout' can still be a subscriber order if is_subscription_start = True

# Data Joining Strategy
# Customer-Order: customers_df.customer_id = orders_df.customer_id
# Customer-Subscription: customers_df.customer_id = subscriptions_df.subscription_customer_user_id
# Order-UTM: orders_df.order_id = orders_utm_df.order_id
# Order-Refund: orders_df.order_id = refunds_df.order_id
# Customer-Address: customers_df.customer_id = customer_shipping_addresses_df.customer_id
# Email-based joins: Match email_hash across customers_df, subscriptions_df, tickets_df, and customer_shipping_addresses_df

# Business Context
# Top Geographic Markets: NY (10.2%), CA (9.3%), FL (9.1%), MA (6.3%), NJ (5.8%)
# Acquisition Channels: Google (34.9%), Direct (29.8%) are primary UTM sources
# Customer Distribution: ~69% cart customers vs ~28% auto-subscribers
# Subscription Status: 57% cancelled, 22.4% on-hold, 14.9% active


# %%
# Import necessary libraries
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set visualization styles
sns.set_style('whitegrid')
plt.style.use('ggplot')

# %%
# Load datasets (including new shipping addresses data)
customers_df = pd.read_csv('../data/customers_redacted.csv')
orders_df = pd.read_csv('../data/orders_redacted.csv')
orders_utm_df = pd.read_csv('../data/orders_with_utm.csv')
quizzes_df = pd.read_csv('../data/quizzes_redacted.csv')
refunds_df = pd.read_csv('../data/refunds_affiliated.csv')
sub_cancel_df = pd.read_csv('../data/subscription_cancellation_reasons.csv')
subscriptions_df = pd.read_csv('../data/subscriptions_redacted.csv')
tickets_df = pd.read_csv('../data/tickets_redacted.csv')
shipping_addresses_df = pd.read_csv('../data/customer_shipping_addresses_redacted.csv')
print("Datasets loaded successfully!")
print(f"New shipping addresses data shape: {shipping_addresses_df.shape}")

# %%
# Convert datetime columns to appropriate format and ensure timezone consistency
date_columns = {
    'orders_df': ['order_created_ts', 'order_created_gmt_ts', 'order_modified_ts',
                  'order_modified_gmt_ts', 'completed_date', 'paid_date'],
    'subscriptions_df': ['subscription_created_ts', 'subscription_last_modified_ts',
                        'subscription_next_payment_ts', 'subscription_cancelled_ts',
                        'subscription_ended_ts'],
    'tickets_df': ['created_at', 'updated_at'],
    'refunds_df': ['refund_date']
}

# Convert all to datetime and strip timezone information
for df_name, cols in date_columns.items():
    for col in cols:
        if col in eval(df_name).columns:
            try:
                eval(df_name)[col] = pd.to_datetime(eval(df_name)[col], errors='coerce')
                if hasattr(eval(df_name)[col].dt, 'tz') and eval(df_name)[col].dt.tz is not None:
                    eval(df_name)[col] = eval(df_name)[col].dt.tz_localize(None)
            except:
                print(f"Could not convert {col} in {df_name}")

print("Date conversions completed!")

# %%
# ===== DEFINE ONE-TIME CUSTOMERS =====
# Based on Colin's clarification (https://northeastern.instructure.com/courses/209854/discussion_topics/2740266), use subscription flags to identify customer types

# Create customer type classification
def classify_customer_type(row):
    """Classify customers based on subscription flags"""
    if row['is_subscription_start'] == True or row['is_subscription_renewal'] == True:
        return 'subscriber'
    else:
        return 'one_time_customer'

# Add customer type to orders
orders_df['customer_type'] = orders_df.apply(classify_customer_type, axis=1)

# Get customer-level classification (if customer ever had a subscription, they're a subscriber)
customer_types = orders_df.groupby('customer_id').agg({
    'customer_type': lambda x: 'subscriber' if 'subscriber' in x.values else 'one_time_customer',
    'order_id': 'count',
    'order_total': 'sum',
    'order_created_ts': ['min', 'max']
}).reset_index()

customer_types.columns = ['customer_id', 'customer_classification', 'total_orders', 
                         'total_spent', 'first_order_date', 'last_order_date']

print("Customer Type Distribution:")
print(customer_types['customer_classification'].value_counts())
print(f"Percentage one-time customers: {customer_types['customer_classification'].value_counts(normalize=True)['one_time_customer']:.1%}")

# %%
# Sample 10 rows of customer types
customer_types.sample(10)

# %%
# ===== FOCUS ON ONE-TIME CUSTOMERS ONLY =====
one_time_customers = customer_types[customer_types['customer_classification'] == 'one_time_customer'].copy()

print(f"\nOne-time customers dataset shape: {one_time_customers.shape}")
print(f"Average orders per one-time customer: {one_time_customers['total_orders'].mean():.2f}")
print(f"Median orders per one-time customer: {one_time_customers['total_orders'].median():.2f}")
print(f"Average spend per one-time customer: ${one_time_customers['total_spent'].mean():.2f}")
print(f"Median spend per one-time customer: ${one_time_customers['total_spent'].median():.2f}")
one_time_customers['avg_days_between_orders'] = (
    (one_time_customers['last_order_date'] - one_time_customers['first_order_date']).dt.days / 
    one_time_customers['total_orders']
)
print(f"Average time between orders for one-time customers: {one_time_customers['avg_days_between_orders'].mean():.2f} days")

# Compare to subscribers
subscribers = customer_types[customer_types['customer_classification'] == 'subscriber'].copy()

print(f"\nSubscribers dataset shape: {subscribers.shape}")
print(f"Average orders per subscriber: {subscribers['total_orders'].mean():.2f}")
print(f"Median orders per subscriber: {subscribers['total_orders'].median():.2f}")
print(f"Average spend per subscriber: ${subscribers['total_spent'].mean():.2f}")
print(f"Median spend per subscriber: ${subscribers['total_spent'].median():.2f}")
subscribers['avg_days_between_orders'] = (
    (subscribers['last_order_date'] - subscribers['first_order_date']).dt.days / 
    subscribers['total_orders']
)
print(f"Average time between orders for subscribers: {subscribers['avg_days_between_orders'].mean():.2f} days")

# %%
# ===== DEFINE CHURN FOR ONE-TIME CUSTOMERS =====
# Problem: Looking from today's perspective creates temporal bias
# Solution: Point-in-time analysis with sufficient observation windows

def create_churn_cohort_analysis(df, observation_window_days=90):
    """
    Create cohort-based churn analysis to avoid temporal bias
    Only include customers who had sufficient time to potentially return
    """
    current_date = datetime.now()
    cutoff_date = current_date - timedelta(days=observation_window_days)
    
    print(f"📅 Cutoff date for analysis: {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"⏱️  Observation window: {observation_window_days} days")
    print(f"   (Only analyzing customers who had {observation_window_days} days to potentially return)")
    
    # Filter to customers whose first purchase was before the cutoff
    # This gives them a fair chance to have returned
    eligible_customers = df[df['first_order_date'] <= cutoff_date].copy()
    
    exclusion_pct = ((len(df) - len(eligible_customers)) / len(df)) * 100
    print(f"\n📊 Dataset Summary:")
    print(f"   Total one-time customers: {len(df):,}")
    print(f"   Eligible for churn analysis: {len(eligible_customers):,}")
    print(f"   Excluded (too recent): {len(df) - len(eligible_customers):,} ({exclusion_pct:.1f}%)")
    
    # Calculate days from first purchase to potential return
    eligible_customers['days_since_first_purchase'] = (
        current_date - eligible_customers['first_order_date']
    ).dt.days
    
    # Define churn and retention for eligible customers only
    eligible_customers['observation_end_date'] = (
        eligible_customers['first_order_date'] + timedelta(days=observation_window_days)
    )
    
    # Key insight: Look at behavior within the observation window
    eligible_customers['is_churned'] = eligible_customers['total_orders'] == 1
    eligible_customers['is_retained'] = eligible_customers['total_orders'] > 1
    
    return eligible_customers

# Apply cohort analysis
one_time_customers_cohort = create_churn_cohort_analysis(one_time_customers, observation_window_days=90)

print(f"\n🎯 Cohort-Based Churn Analysis Results:")
print(f"   Churned customers: {one_time_customers_cohort['is_churned'].sum():,} ({one_time_customers_cohort['is_churned'].mean():.1%})")
print(f"   Retained customers: {one_time_customers_cohort['is_retained'].sum():,} ({one_time_customers_cohort['is_retained'].mean():.1%})")

# ===== TEMPORAL TREND ANALYSIS =====
def analyze_churn_trends_over_time(df, window_days=90):
    """
    Analyze how churn rates change over time by sampling different evaluation dates
    Enhanced with median spend tracking
    """
    # Get date range for analysis
    min_date = df['first_order_date'].min()
    max_date = df['first_order_date'].max() - timedelta(days=window_days)  # Leave room for observation
    
    print(f"\n⏰ Temporal Analysis Setup:")
    print(f"   Analysis period: {min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}")
    print(f"   Creating monthly cohorts with {window_days}-day observation windows")
    
    # Create monthly evaluation points
    eval_dates = pd.date_range(start=min_date, end=max_date, freq='M')
    
    trend_results = []
    
    for eval_date in eval_dates:
        # Get customers who made first purchase before this eval date
        cohort_start = eval_date - timedelta(days=30)  # Monthly cohort
        cohort_customers = df[
            (df['first_order_date'] >= cohort_start) & 
            (df['first_order_date'] < eval_date)
        ]
        
        if len(cohort_customers) < 50:  # Skip small cohorts
            continue
            
        # Check their behavior within the observation window
        observation_end = eval_date + timedelta(days=window_days)
        
        # For each customer, check if they made additional purchases within window
        cohort_churn_rate = (cohort_customers['total_orders'] == 1).mean()
        cohort_retention_rate = (cohort_customers['total_orders'] > 1).mean()
        
        trend_results.append({
            'cohort_month': eval_date.strftime('%Y-%m'),
            'cohort_size': len(cohort_customers),
            'churn_rate': cohort_churn_rate,
            'retention_rate': cohort_retention_rate,
            'avg_orders': cohort_customers['total_orders'].mean(),
            'avg_spend': cohort_customers['total_spent'].mean(),
            'median_spend': cohort_customers['total_spent'].median()  # Added median spend
        })
    
    return pd.DataFrame(trend_results)

# Analyze trends over time
churn_trends = analyze_churn_trends_over_time(one_time_customers, window_days=90)

print(f"\n📈 Temporal Trend Analysis Results:")
print(f"   Analyzed cohorts: {len(churn_trends)}")
if len(churn_trends) > 0:
    print(f"   Average churn rate across all periods: {churn_trends['churn_rate'].mean():.1%}")
    print(f"   Churn rate range: {churn_trends['churn_rate'].min():.1%} - {churn_trends['churn_rate'].max():.1%}")
    print(f"   Churn rate volatility (std dev): {churn_trends['churn_rate'].std():.1%}")

# ===== ENHANCED VISUALIZATION OF TEMPORAL TRENDS =====
if len(churn_trends) > 5:  # Only plot if we have enough data points
    
    # Calculate trend direction for context
    recent_churn = churn_trends.tail(6)['churn_rate'].mean()  # Last 6 months
    early_churn = churn_trends.head(6)['churn_rate'].mean()   # First 6 months
    trend_direction = "📈 INCREASING" if recent_churn > early_churn else "📉 DECREASING"
    trend_magnitude = abs(recent_churn - early_churn)
    
    print(f"\n🔍 Key Trend Insights:")
    print(f"   Early period churn (first 6 months): {early_churn:.1%}")
    print(f"   Recent period churn (last 6 months): {recent_churn:.1%}")
    print(f"   Overall trend: {trend_direction} by {trend_magnitude:.1%}")
    
    # Create improved visualization (3 plots instead of 4)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Churn rate over time with trend line
    dates = pd.to_datetime(churn_trends['cohort_month'])
    ax1.plot(dates, churn_trends['churn_rate'], marker='o', linewidth=2, markersize=6, color='red', alpha=0.8)
    
    # Add trend line
    z = np.polyfit(range(len(churn_trends)), churn_trends['churn_rate'], 1)
    p = np.poly1d(z)
    ax1.plot(dates, p(range(len(churn_trends))), "--", alpha=0.6, color='darkred', linewidth=2, label=f'Trend ({trend_direction})')
    
    ax1.set_title('🚨 Churn Rate Over Time\n(Monthly Cohorts with 90-day Observation)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Churn Rate', fontsize=11)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format y-axis as percentage
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Plot 2: Average vs Median Spend Analysis
    ax2.plot(dates, churn_trends['avg_spend'], marker='s', linewidth=2, label='Average Spend', color='green', markersize=5)
    ax2.plot(dates, churn_trends['median_spend'], marker='^', linewidth=2, label='Median Spend', color='darkgreen', markersize=5, linestyle='--')
    
    ax2.set_title('💰 Customer Spend Trends\n(Average vs Median)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Spend Amount ($)', fontsize=11)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Format y-axis as currency
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '${:.0f}'.format(y)))
    
    # Plot 3: Cohort Size Over Time (Bar chart for better readability)
    bars = ax3.bar(dates, churn_trends['cohort_size'], alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.5)
    
    # Add value labels on top of bars for key periods
    for i, (date, size) in enumerate(zip(dates, churn_trends['cohort_size'])):
        if i % 6 == 0:  # Label every 6th bar to avoid crowding
            ax3.text(date, size + max(churn_trends['cohort_size']) * 0.01, f'{int(size)}', 
                    ha='center', va='bottom', fontsize=9)
    
    ax3.set_title('👥 Cohort Size Trends\n(Customer Volume by Month)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Customers', fontsize=11)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Enhanced trend insights with correlation analysis
    print(f"\n🔗 Correlation Analysis:")
    spend_churn_corr = churn_trends['avg_spend'].corr(churn_trends['churn_rate'])
    size_churn_corr = churn_trends['cohort_size'].corr(churn_trends['churn_rate'])
    spend_size_corr = churn_trends['avg_spend'].corr(churn_trends['cohort_size'])
    
    print(f"   Average Spend ↔ Churn Rate: {spend_churn_corr:.3f}")
    if abs(spend_churn_corr) > 0.3:
        direction = "Higher spend = Lower churn" if spend_churn_corr < 0 else "Higher spend = Higher churn"
        print(f"     💡 {direction} (moderate correlation)")
    
    print(f"   Cohort Size ↔ Churn Rate: {size_churn_corr:.3f}")
    if abs(size_churn_corr) > 0.3:
        direction = "Larger cohorts = Lower churn" if size_churn_corr < 0 else "Larger cohorts = Higher churn" 
        print(f"     💡 {direction} (moderate correlation)")
    
    print(f"   Average Spend ↔ Cohort Size: {spend_size_corr:.3f}")
    
    # Identify most problematic periods
    worst_periods = churn_trends.nlargest(3, 'churn_rate')[['cohort_month', 'churn_rate', 'cohort_size', 'avg_spend']]
    print(f"\n⚠️  Worst Performing Periods (Highest Churn):")
    for _, period in worst_periods.iterrows():
        print(f"   {period['cohort_month']}: {period['churn_rate']:.1%} churn, {int(period['cohort_size'])} customers, ${period['avg_spend']:.0f} avg spend")

else:
    print("❌ Insufficient data points for visualization (need > 5 cohorts)")

# Update the dataset for modeling to use cohort-based approach
one_time_customers = one_time_customers_cohort.copy()

print(f"\n✅ Analysis Complete - Dataset updated for further modeling")
print(f"   Final dataset size: {len(one_time_customers):,} customers")
print(f"   Date range: {one_time_customers['first_order_date'].min().strftime('%Y-%m')} to {one_time_customers['first_order_date'].max().strftime('%Y-%m')}")


