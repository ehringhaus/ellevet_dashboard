# %%
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import requests
from google.colab import drive
import os

import numpy as np # linear algebra
import pandas as pd


# %%
drive.mount('/content/drive')
data_path = '/content/drive/My Drive/ALY6980/'

#file_name = 'orders_with_utm.csv'
#orders_with_utm = pd.read_csv(data_path + file_name)

file_name = 'subscription_cancellation_reasons.csv'
cancellation_reasons = pd.read_csv(data_path + file_name)

file_name = 'quizzes_redacted.csv'
quizzes = pd.read_csv(data_path + file_name)

# %%
file_name = 'tickets_redacted.csv'
tickets_redacted = pd.read_csv(data_path + file_name)

file_name = 'refunds_affiliated.csv'
refunds = pd.read_csv(data_path + file_name)

file_name = 'quizzes_redacted.csv'
quizzes = pd.read_csv(data_path + file_name)

# %%
file_name = 'subscriptions_redacted.csv'
subscriptions = pd.read_csv(data_path + file_name)

file_name = 'customers_redacted.csv'
customers = pd.read_csv(data_path + file_name)

file_name = 'orders_redacted.csv'
orders = pd.read_csv(data_path + file_name)

# %% [markdown]
# subscription_parent_id', 'subscription_customer_user_id

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Convert date columns to datetime objects
orders['order_created_ts'] = pd.to_datetime(orders['order_created_ts'])
subscriptions['subscription_created_ts'] = pd.to_datetime(subscriptions['subscription_created_ts'])

# Extract month and year
orders['month'] = orders['order_created_ts'].dt.to_period('M')
subscriptions['month'] = subscriptions['subscription_created_ts'].dt.to_period('M')

# Count orders and subscriptions by month
orders_by_month = orders.groupby('month').size().reset_index(name='order_count')
subscriptions_by_month = subscriptions.groupby('month').size().reset_index(name='subscription_count')

# Merge the dataframes
seasonal_trends = pd.merge(orders_by_month, subscriptions_by_month, on='month', how='outer').fillna(0)

# Convert 'month' back to datetime for plotting
seasonal_trends['month'] = seasonal_trends['month'].dt.to_timestamp()

# Plotting the seasonal trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=seasonal_trends, x='month', y='order_count', label='Orders')
sns.lineplot(data=seasonal_trends, x='month', y='subscription_count', label='Subscriptions')
plt.title('Seasonal Trends in Orders vs Subscriptions')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# %%
orders.columns

# %%
subscriptions.columns

# %%
# Add rolling averages (3-month window)
seasonal_trends['orders_rolling_avg'] = seasonal_trends['order_count'].rolling(window=3, center=True).mean()
seasonal_trends['subscriptions_rolling_avg'] = seasonal_trends['subscription_count'].rolling(window=3, center=True).mean()

# Plot with rolling average
plt.figure(figsize=(12, 6))
sns.lineplot(data=seasonal_trends, x='month', y='orders_rolling_avg', label='Orders (3-mo avg)', linestyle='--')
sns.lineplot(data=seasonal_trends, x='month', y='subscriptions_rolling_avg', label='Subscriptions (3-mo avg)', linestyle='--')
plt.title('Smoothed Seasonal Trends (3-month Rolling Avg)')
plt.xlabel('Month')
plt.ylabel('Smoothed Count')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# %%
from statsmodels.tsa.seasonal import STL

# Ensure monthly datetime index for decomposition
seasonal_trends.set_index('month', inplace=True)

# Apply STL decomposition
orders_stl = STL(seasonal_trends['order_count'], period=12).fit()
subs_stl = STL(seasonal_trends['subscription_count'], period=12).fit()

# Plot decomposition for Orders
orders_stl.plot()
plt.suptitle('STL Decomposition - Orders', fontsize=14)
plt.tight_layout()
plt.show()

# Plot decomposition for Subscriptions
subs_stl.plot()
plt.suptitle('STL Decomposition - Subscriptions', fontsize=14)
plt.tight_layout()
plt.show()

# Reset index for future plotting
seasonal_trends.reset_index(inplace=True)

# %% [markdown]
# The number of orders grew steadily from 2018 through 2023, with minor seasonal fluctuations likely tied to sales cycles or customer behavior. In late 2023 and early 2024, growth plateaued and dipped sharply in early 2025—possibly due to external disruptions. The residuals suggest the model captured most of the structure, but some unexpected events still occurred.
# 
# 

# %%
# General libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Clustering and decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import umap.umap_ as umap

# Time series decomposition
from statsmodels.tsa.seasonal import STL

# Warnings and display
import warnings
warnings.filterwarnings('ignore')


# %%
# Select numerical features for clustering
feature_columns = ['order_count', 'subscription_count']  # update with your actual features
scaled_features = StandardScaler().fit_transform(seasonal_trends[feature_columns])

# PCA
pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(scaled_features)
seasonal_trends['pca_1'] = pca_components[:, 0]
seasonal_trends['pca_2'] = pca_components[:, 1]

# UMAP
reducer = umap.UMAP(random_state=42)
umap_components = reducer.fit_transform(scaled_features)
seasonal_trends['umap_1'] = umap_components[:, 0]
seasonal_trends['umap_2'] = umap_components[:, 1]


# %%
def perform_kmeans_and_score(embedded_data, max_k=10, method_name='PCA'):
    scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embedded_data)
        score = silhouette_score(embedded_data, labels)
        scores.append((k, score))
        print(f"{method_name} | k={k} | Silhouette Score={score:.3f}")
    return scores

# Run for PCA and UMAP
pca_scores = perform_kmeans_and_score(pca_components, method_name='PCA')
umap_scores = perform_kmeans_and_score(umap_components, method_name='UMAP')

# %%
# Final clustering (update `n_clusters` if needed)
best_k = 4

# PCA Clustering
kmeans_pca = KMeans(n_clusters=best_k, random_state=42)
seasonal_trends['pca_cluster'] = kmeans_pca.fit_predict(pca_components)

# UMAP Clustering
kmeans_umap = KMeans(n_clusters=best_k, random_state=42)
seasonal_trends['umap_cluster'] = kmeans_umap.fit_predict(umap_components)


# %%
# Plot clusters in PCA space
plt.figure(figsize=(12, 5))
sns.scatterplot(x='pca_1', y='pca_2', hue='pca_cluster', palette='tab10', data=seasonal_trends)
plt.title('PCA Clusters')
plt.show()

# Plot clusters in UMAP space
plt.figure(figsize=(12, 5))
sns.scatterplot(x='umap_1', y='umap_2', hue='umap_cluster', palette='tab10', data=seasonal_trends)
plt.title('UMAP Clusters')
plt.show()


# %%
seasonal_trends.head()

# %%
# Convert date column to datetime if not already
seasonal_trends['month'] = pd.to_datetime(seasonal_trends['month'])

# Set time index if needed
seasonal_trends = seasonal_trends.set_index('month')

# Resample and plot per cluster
def plot_time_series_by_cluster(df, cluster_col, value_col, title_prefix):
    grouped = df.groupby([pd.Grouper(freq='M'), cluster_col])[value_col].mean().reset_index()
    plt.figure(figsize=(12, 6))
    for cluster in sorted(df[cluster_col].unique()):
        cluster_data = grouped[grouped[cluster_col] == cluster]
        plt.plot(cluster_data['month'], cluster_data[value_col], label=f"Cluster {cluster}")
    plt.title(f"{title_prefix} - Avg {value_col} Over Time")
    plt.xlabel("Date")
    plt.ylabel(value_col)
    plt.legend()
    plt.show()

# PCA clusters
plot_time_series_by_cluster(seasonal_trends, 'pca_cluster', 'order_count', 'PCA Clusters')
plot_time_series_by_cluster(seasonal_trends, 'pca_cluster', 'subscription_count', 'PCA Clusters')

# UMAP clusters
plot_time_series_by_cluster(seasonal_trends, 'umap_cluster', 'order_count', 'UMAP Clusters')
plot_time_series_by_cluster(seasonal_trends, 'umap_cluster', 'subscription_count', 'UMAP Clusters')


# %%
# Add back feature values to cluster labels
feature_df = pd.DataFrame(scaled_features, columns=feature_columns)
feature_df['pca_cluster'] = seasonal_trends['pca_cluster'].values
feature_df['umap_cluster'] = seasonal_trends['umap_cluster'].values


# %%
# Ensure the 'month' columns are consistent
orders['month'] = orders['order_created_ts'].dt.to_period('M').dt.to_timestamp()
subscriptions['month'] = subscriptions['subscription_created_ts'].dt.to_period('M').dt.to_timestamp()

# Reset index to expose 'month' for merging
clusters_to_merge = seasonal_trends.reset_index()[['month', 'pca_cluster', 'umap_cluster']]

# Merge with orders
orders_clustered = pd.merge(orders, clusters_to_merge, on='month', how='left')

# Merge with subscriptions
subscriptions_clustered = pd.merge(subscriptions, clusters_to_merge, on='month', how='left')


# %%
# Preview a few rows to confirm the merge worked
print(orders_clustered[['month', 'order_created_ts', 'pca_cluster', 'umap_cluster']].head())
print(subscriptions_clustered[['month', 'subscription_created_ts', 'pca_cluster', 'umap_cluster']].head())

# Group by UMAP and PCA clusters to examine trends
orders_summary = orders_clustered.groupby('umap_cluster').agg({
    'order_id': 'count',
    'customer_id': pd.Series.nunique,
    'order_total': 'mean'
}).rename(columns={'order_id': 'total_order_count', 'customer_id': 'unique_customers', 'order_value': 'avg_order_value'})

subscriptions_summary = subscriptions_clustered.groupby('umap_cluster').agg({
    'subscription_id': 'count',
    'subscription_customer_user_id': pd.Series.nunique,
    'subscription_order_total': 'mean',
    'subscription_status': pd.Series.nunique
}).rename(columns={'subscription_id': 'total_subscriptions', 'customer_id': 'unique_customers', 'subscription_order_total':'avg_sub_order_value','plan_type': 'distinct_plan_types'})

# Display summaries
#print("\n📦 Orders Summary by UMAP Cluster:")
##print(orders_summary)

#print("\n🔄 Subscriptions Summary by UMAP Cluster:")
#print(subscriptions_summary)


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Bar plots for Orders - UMAP clusters
plt.figure(figsize=(8, 5))
sns.countplot(data=orders_clustered, x='umap_cluster', palette='Blues')
plt.title('Order Count per UMAP Cluster')
plt.xlabel('UMAP Cluster')
plt.ylabel('Number of Orders')
plt.tight_layout()
plt.show()

# Bar plots for Orders - PCA clusters
plt.figure(figsize=(8, 5))
sns.countplot(data=orders_clustered, x='pca_cluster', palette='Purples')
plt.title('Order Count per PCA Cluster')
plt.xlabel('PCA Cluster')
plt.ylabel('Number of Orders')
plt.tight_layout()
plt.show()

# Subscriptions - UMAP clusters
plt.figure(figsize=(8, 5))
sns.countplot(data=subscriptions_clustered, x='umap_cluster', palette='Greens')
plt.title('Subscription Count per UMAP Cluster')
plt.xlabel('UMAP Cluster')
plt.ylabel('Number of Subscriptions')
plt.tight_layout()
plt.show()

# Subscriptions - PCA clusters
plt.figure(figsize=(8, 5))
sns.countplot(data=subscriptions_clustered, x='pca_cluster', palette='Oranges')
plt.title('Subscription Count per PCA Cluster')
plt.xlabel('PCA Cluster')
plt.ylabel('Number of Subscriptions')
plt.tight_layout()
plt.show()


# %%
# Group orders by month and cluster
orders_time_umap = orders_clustered.groupby(['month', 'umap_cluster']).size().reset_index(name='order_count')
orders_time_pca = orders_clustered.groupby(['month', 'pca_cluster']).size().reset_index(name='order_count')

# Plot UMAP clusters over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=orders_time_umap, x='month', y='order_count', hue='umap_cluster', palette='tab10')
plt.title('Order Trends Over Time by UMAP Cluster')
plt.xlabel('Month')
plt.ylabel('Order Count')
plt.xticks(rotation=45)
plt.legend(title='UMAP Cluster')
plt.tight_layout()
plt.show()

# Plot PCA clusters over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=orders_time_pca, x='month', y='order_count', hue='pca_cluster', palette='tab20')
plt.title('Order Trends Over Time by PCA Cluster')
plt.xlabel('Month')
plt.ylabel('Order Count')
plt.xticks(rotation=45)
plt.legend(title='PCA Cluster')
plt.tight_layout()
plt.show()


# %%
# Group subscriptions by month and cluster
subs_time_umap = subscriptions_clustered.groupby(['month', 'umap_cluster']).size().reset_index(name='subscription_count')
subs_time_pca = subscriptions_clustered.groupby(['month', 'pca_cluster']).size().reset_index(name='subscription_count')

# Plot UMAP clusters over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=subs_time_umap, x='month', y='subscription_count', hue='umap_cluster', palette='tab10')
plt.title('Subscription Trends Over Time by UMAP Cluster')
plt.xlabel('Month')
plt.ylabel('Subscription Count')
plt.xticks(rotation=45)
plt.legend(title='UMAP Cluster')
plt.tight_layout()
plt.show()

# Plot PCA clusters over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=subs_time_pca, x='month', y='subscription_count', hue='pca_cluster', palette='tab20')
plt.title('Subscription Trends Over Time by PCA Cluster')
plt.xlabel('Month')
plt.ylabel('Subscription Count')
plt.xticks(rotation=45)
plt.legend(title='PCA Cluster')
plt.tight_layout()
plt.show()


# %%
from statsmodels.tsa.seasonal import STL

# Ensure month is datetime and set as index
orders_clustered['month'] = pd.to_datetime(orders_clustered['month'])
subscriptions_clustered['month'] = pd.to_datetime(subscriptions_clustered['month'])

# Function for STL decomposition plotting
def plot_stl_for_cluster(df, date_col, count_col, cluster_col, cluster_id, entity='Orders'):
    cluster_df = df[df[cluster_col] == cluster_id]
    ts = cluster_df.groupby(date_col).size()
    ts = ts.asfreq('MS').fillna(0)  # Ensure monthly freq

    stl = STL(ts, period=12)
    result = stl.fit()

    result.plot()
    plt.suptitle(f'STL Decomposition - {entity} Cluster {cluster_id}', fontsize=14)
    plt.tight_layout()
    plt.show()

# Run for UMAP clusters (Orders)
for cluster in sorted(orders_clustered['umap_cluster'].unique()):
    plot_stl_for_cluster(orders_clustered, 'month', 'order_id', 'umap_cluster', cluster, entity='Orders')

# Run for PCA clusters (Orders)
for cluster in sorted(orders_clustered['pca_cluster'].unique()):
    plot_stl_for_cluster(orders_clustered, 'month', 'order_id', 'pca_cluster', cluster, entity='Orders')

# Repeat for Subscriptions
for cluster in sorted(subscriptions_clustered['umap_cluster'].unique()):
    plot_stl_for_cluster(subscriptions_clustered, 'month', 'subscription_id', 'umap_cluster', cluster, entity='Subscriptions')

for cluster in sorted(subscriptions_clustered['pca_cluster'].unique()):
    plot_stl_for_cluster(subscriptions_clustered, 'month', 'subscription_id', 'pca_cluster', cluster, entity='Subscriptions')


# %%
# First, group by month and cluster (UMAP)
orders_clustered['month'] = orders_clustered['order_created_ts'].dt.to_period('M')
order_value_umap = orders_clustered.groupby(['month', 'umap_cluster'])['order_total'].mean().reset_index()
order_value_umap['month'] = order_value_umap['month'].dt.to_timestamp()

# PCA clusters
order_value_pca = orders_clustered.groupby(['month', 'pca_cluster'])['order_total'].mean().reset_index()
order_value_pca['month'] = order_value_pca['month'].dt.to_timestamp()


# %%
# First, group by month and cluster (UMAP)
subscriptions_clustered['month'] = subscriptions_clustered['subscription_created_ts'].dt.to_period('M')
subs_value_umap = subscriptions_clustered.groupby(['month', 'umap_cluster'])['subscription_order_total'].mean().reset_index()
subs_value_umap['month'] = subs_value_umap['month'].dt.to_timestamp()

# PCA clusters
subs_value_pca = subscriptions_clustered.groupby(['month', 'pca_cluster'])['subscription_order_total'].mean().reset_index()
subs_value_pca['month'] = subs_value_pca['month'].dt.to_timestamp()

# %%
# UMAP clusters (Orders)
plt.figure(figsize=(12, 6))
sns.lineplot(data=order_value_umap, x='month', y='order_total', hue='umap_cluster', palette='tab10')
plt.title('Average Order Value Over Time by UMAP Cluster (Orders)')
plt.xlabel('Month')
plt.ylabel('Average Order Value')
plt.xticks(rotation=45)
plt.legend(title='UMAP Cluster')
plt.tight_layout()
plt.show()

# PCA clusters (Orders)
plt.figure(figsize=(12, 6))
sns.lineplot(data=order_value_pca, x='month', y='order_total', hue='pca_cluster', palette='tab20')
plt.title('Average Order Value Over Time by PCA Cluster (Orders)')
plt.xlabel('Month')
plt.ylabel('Average Order Value')
plt.xticks(rotation=45)
plt.legend(title='PCA Cluster')
plt.tight_layout()
plt.show()


# %%
# UMAP clusters (Subscriptions)
plt.figure(figsize=(12, 6))
sns.lineplot(data=subs_value_umap, x='month', y='subscription_order_total', hue='umap_cluster', palette='Set2')
plt.title('Average Subscription Value Over Time by UMAP Cluster')
plt.xlabel('Month')
plt.ylabel('Average Subscription Value')
plt.xticks(rotation=45)
plt.legend(title='UMAP Cluster')
plt.tight_layout()
plt.show()

# PCA clusters (Subscriptions)
plt.figure(figsize=(12, 6))
sns.lineplot(data=subs_value_pca, x='month', y='subscription_order_total', hue='pca_cluster', palette='Set3')
plt.title('Average Subscription Value Over Time by PCA Cluster')
plt.xlabel('Month')
plt.ylabel('Average Subscription Value')
plt.xticks(rotation=45)
plt.legend(title='PCA Cluster')
plt.tight_layout()
plt.show()


# %%
import ast
import seaborn as sns
import matplotlib.pyplot as plt

# STEP 1: Parse line_items and explode
orders_clustered['parsed_line_items'] = orders_clustered['line_items'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

orders_exploded = orders_clustered.explode('parsed_line_items').reset_index(drop=True)

# Extract relevant product details
orders_exploded['product_id'] = orders_exploded['parsed_line_items'].apply(
    lambda x: x.get('product_id') if isinstance(x, dict) else None
)
orders_exploded['product_name'] = orders_exploded['parsed_line_items'].apply(
    lambda x: x.get('product_name') if isinstance(x, dict) else None
)
orders_exploded['line_total'] = orders_exploded['parsed_line_items'].apply(
    lambda x: x.get('line_total') if isinstance(x, dict) else None
)

# %%
umap_summary = orders_exploded.groupby(['umap_cluster', 'product_name']).agg(
    order_count=('order_id', 'count'),
    product_count=('product_id', 'count'),
    total_revenue=('line_total', 'sum'),
    avg_order_value=('line_total', 'mean')
).reset_index()

# %%
pca_summary = orders_exploded.groupby(['pca_cluster', 'product_name']).agg(
    order_count=('order_id', 'count'),
    product_count=('product_id', 'count'),
    total_revenue=('line_total', 'sum'),
    avg_order_value=('line_total', 'mean')
).reset_index()

# %%
for cluster in sorted(pca_summary['pca_cluster'].unique()):
    top_products = pca_summary[pca_summary['pca_cluster'] == cluster].nlargest(5, 'product_count')

    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_products, y='product_name', x='product_count', palette='magma')
    plt.title(f'Top Products by Count - PCA Cluster {cluster}')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Total Count")
    plt.tight_layout()
    plt.show()

# %%
import plotly.express as px
import plotly.graph_objects as go

# Step 1: Same as before — group by role, cluster, product
umap_summary = orders_exploded.groupby(
    ['umap_cluster', 'product_name', 'customer_role']
).agg(
    product_count=('product_id', 'count'),
    total_revenue=('line_total', 'sum'),
    avg_order_value=('line_total', 'mean')
).reset_index()

# Step 2: Prepare data for each combination of role + cluster
fig = go.Figure()

roles = umap_summary['customer_role'].dropna().unique()
clusters = sorted(umap_summary['umap_cluster'].dropna().unique())

# Store visibility map
visibility = []
buttons = []

# Add traces (one per role + cluster combo)
for role in roles:
    for cluster in clusters:
        filtered = (
            umap_summary[
                (umap_summary['customer_role'] == role) &
                (umap_summary['umap_cluster'] == cluster)
            ]
            .nlargest(5, 'product_count')
        )
        trace = go.Bar(
            x=filtered['product_count'],
            y=filtered['product_name'],
            name=f'{role.capitalize()} - Cluster {cluster}',
            orientation='h',
            visible=False
        )
        fig.add_trace(trace)

# Set initial visibility
default_index = 0
fig.data[default_index].visible = True

# Step 3: Add interactive dropdown menu
for i, role in enumerate(roles):
    for j, cluster in enumerate(clusters):
        label = f'{role.capitalize()} - Cluster {cluster}'
        index = i * len(clusters) + j
        vis = [False] * len(fig.data)
        vis[index] = True
        buttons.append(dict(label=label, method='update', args=[{'visible': vis}, {'title': label}]))

fig.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        x=1.1,
        y=1,
        xanchor='left',
        yanchor='top'
    )],
    title='Top Products by UMAP Cluster and Customer Role',
    xaxis_title='Product Count',
    yaxis_title='Product Name',
    height=500,
    width=900
)

fig.show()


# %%
# For orders
orders_by_state_umap = orders_clustered.groupby(['shipping_state', 'umap_cluster']).size().reset_index(name='order_count')
orders_by_state_pca = orders_clustered.groupby(['shipping_state', 'pca_cluster']).size().reset_index(name='order_count')

# For subscriptions
#subs_by_state_umap = subscriptions_clustered.groupby(['shipping_state', 'umap_cluster']).size().reset_index(name='subscription_count')
#subs_by_state_pca = subscriptions_clustered.groupby(['shipping_state', 'pca_cluster']).size().reset_index(name='subscription_count')



# %% [markdown]
# redo the shipping by doing a join where you join on order id and then use the true or false category to doublecheck that the match is correct

# %%
import plotly.express as px

fig = px.choropleth(
    orders_by_state_umap,
    locations='shipping_state',  # must be the 2-letter state code
    locationmode='USA-states',
    color='order_count',
    scope='usa',
    animation_frame='umap_cluster',  # one frame per cluster
    color_continuous_scale='Viridis',
    title='Order Count by State (UMAP Clusters)'
)
fig.show()


# %%
# For orders
orders_by_state_umap_role = orders_clustered.groupby(['shipping_state', 'umap_cluster', 'customer_role']).size().reset_index(name='order_count')

# For subscriptions
#subs_by_state_umap_role = subscriptions_clustered.groupby(['shipping_state', 'umap_cluster', 'customer_role']).size().reset_index(name='subscription_count')


# %%
import plotly.express as px

fig = px.choropleth(
    orders_by_state_umap_role,
    locations='shipping_state',
    locationmode='USA-states',
    color='order_count',
    scope='usa',
    animation_frame='umap_cluster',     # One frame per cluster
    facet_col='customer_role',          # Separate plots per role
    color_continuous_scale='Plasma',
    title='UMAP Clustered Orders by Customer Role and State'
)
fig.show()

# %%
import pandas as pd
import ast

def safe_parse(x):
    if pd.isna(x):
        return []  # Return empty list for NaN
    try:
        return ast.literal_eval(x)
    except Exception:
        return []  # If parsing fails, treat it as empty

# Make a copy and parse safely
orders_exploded = orders_clustered.copy()
orders_exploded['line_items'] = orders_exploded['line_items'].apply(safe_parse)

# Explode into multiple rows (one per product)
orders_exploded = orders_exploded.explode('line_items')

# Extract product name and quantity safely
orders_exploded['product_name'] = orders_exploded['line_items'].apply(
    lambda x: x.get('product_name') if isinstance(x, dict) else None
)
orders_exploded['product_quantity'] = orders_exploded['line_items'].apply(
    lambda x: x.get('quantity') if isinstance(x, dict) else 0
)


# %%
# Group by state, cluster, and product
product_geo_cluster = orders_exploded.groupby(
    ['shipping_state', 'umap_cluster', 'product_name']
)['product_quantity'].sum().reset_index()

# %%
# Optional: focus on top 10 products
top_products = product_geo_cluster.groupby('product_name')['product_quantity'].sum().nlargest(10).index
filtered_data = product_geo_cluster[product_geo_cluster['product_name'].isin(top_products)]


# %%
import plotly.express as px

fig = px.choropleth(
    filtered_data,
    locations='shipping_state',
    locationmode='USA-states',
    color='product_quantity',
    animation_frame='product_name',
    facet_col='umap_cluster',
    scope='usa',
    color_continuous_scale='Viridis',
    title='Top Product Volume by UMAP Cluster and State'
)
fig.show()

# %%
orders_exploded.columns

# %%
# Sort and calculate recency
orders_clustered['order_created_ts'] = pd.to_datetime(orders_clustered['order_created_ts'])
latest_order = orders_clustered.groupby('customer_id')['order_created_ts'].max().reset_index()
latest_order['days_since_last_order'] = (orders_clustered['order_created_ts'].max() - latest_order['order_created_ts']).dt.days
latest_order['churned'] = latest_order['days_since_last_order'] > 180 # customize this threshold

# Merge back to orders_clustered
orders_clustered = orders_clustered.merge(latest_order[['customer_id', 'churned']], on='customer_id', how='left')


# %%
churn_by_cluster_orders = orders_clustered.groupby('umap_cluster')['churned'].mean().reset_index()
churn_by_cluster_orders['churn_rate'] = churn_by_cluster_orders['churned'] * 100

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
sns.barplot(data=churn_by_cluster_orders, x='umap_cluster', y='churn_rate', palette='viridis')
plt.title('Churn Rate by UMAP Cluster')
plt.ylabel('Churn Rate (%)')
plt.xlabel('UMAP Cluster')
plt.tight_layout()
plt.show()

#break it down by Subscribers and Customers
#plt.figure(figsize=(10, 5))
#sns.barplot(data=churn_by_cluster_orders, x='umap_cluster', #y='churn_rate', palette='magma')
#plt.title('Churn Rate by UMAP Cluster – Orders')
#plt.ylabel('Churn Rate (%)')
#plt.xlabel('UMAP Cluster')
#plt.tight_layout()
#plt.show()

# %%
# Ensure datetime format
orders_clustered['order_created_ts'] = pd.to_datetime(orders_clustered['order_created_ts'])

# Extract month (period) and assign
orders_clustered['order_month'] = orders_clustered['order_created_ts'].dt.to_period('M')

# Get latest cluster per customer per month
monthly_cluster_assignment = (
    orders_clustered.sort_values(['customer_id', 'order_created_ts'])  # sort for latest order per month
    .groupby(['customer_id', 'order_month'])[['pca_cluster', 'umap_cluster']]  # cluster type of interest
    .last()  # gets the most recent order in the month
    .reset_index()
)

# Optional: Pivot to track cluster over time per customer (good for heatmaps or transition graphs)
pivot_umap = monthly_cluster_assignment.pivot(index='customer_id', columns='order_month', values='umap_cluster')
pivot_pca = monthly_cluster_assignment.pivot(index='customer_id', columns='order_month', values='pca_cluster')

# Preview
print("📊 UMAP Cluster Over Time (per customer):")
display(pivot_umap.head())

print("📊 PCA Cluster Over Time (per customer):")
display(pivot_pca.head())


# %%
# Churn rate by cluster and month
churned_trend = (
    orders_clustered.groupby(['order_month', 'umap_cluster'])
    .agg(churn_rate=('churned', 'mean'), total_customers=('customer_id', 'nunique'))
    .reset_index()
)

churned_trend['order_month'] = churned_trend['order_month'].dt.to_timestamp()
churned_trend['churn_rate'] *= 100  # To percent

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=churned_trend, x='order_month', y='churn_rate', hue='umap_cluster', palette='coolwarm')
plt.title('📉 Churn Rate Over Time by UMAP Cluster')
plt.xlabel('Month')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=45)
plt.legend(title='UMAP Cluster')
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval

# ========== 1. PREPARE BASE DATA ==========
orders_clustered['order_created_ts'] = pd.to_datetime(orders_clustered['order_created_ts'])
orders_clustered['order_month'] = orders_clustered['order_created_ts'].dt.to_period('M')
orders_clustered['order_year'] = orders_clustered['order_created_ts'].dt.year

# Filter only relevant customer roles
orders_clustered = orders_clustered[orders_clustered['customer_role'].isin(['customer', 'subscriber'])]

# ========== 2. CALCULATE CHURN ==========
# Get latest order per customer + role
latest_order = (
    orders_clustered.groupby(['customer_id', 'customer_role'])['order_created_ts']
    .max()
    .reset_index(name='last_order_date')
)

# Define churn as >90 days since last order
reference_date = orders_clustered['order_created_ts'].max()
latest_order['days_since_last_order'] = (reference_date - latest_order['last_order_date']).dt.days
latest_order['churned'] = latest_order['days_since_last_order'] > 90

# Drop any existing churn column to avoid merge conflict
orders_clustered = orders_clustered.drop(columns=['churned'], errors='ignore')

# Merge churn flag back into orders_clustered
orders_clustered = orders_clustered.merge(
    latest_order[['customer_id', 'customer_role', 'churned']],
    on=['customer_id', 'customer_role'],
    how='left'
)

# ========== 3. CLUSTER ASSIGNMENTS BY MONTH ==========
monthly_cluster_assignment = (
    orders_clustered.sort_values(['customer_id', 'order_created_ts'])
    .groupby(['customer_id', 'customer_role', 'order_month'])[['pca_cluster', 'umap_cluster']]
    .last()
    .reset_index()
)

pivot_umap = monthly_cluster_assignment.pivot(index='customer_id', columns='order_month', values='umap_cluster')
pivot_pca = monthly_cluster_assignment.pivot(index='customer_id', columns='order_month', values='pca_cluster')

# ========== 4. EXPLODE LINE ITEMS TO GET PRODUCT INFO ==========
# Safely convert line_items strings to list of dicts
orders_clustered['line_items'] = orders_clustered['line_items'].apply(
    lambda x: literal_eval(x) if isinstance(x, str) else x
)

orders_exploded = orders_clustered.explode('line_items')
orders_exploded = orders_exploded[orders_exploded['line_items'].notnull()]
orders_exploded['product_name'] = orders_exploded['line_items'].apply(
    lambda x: x.get('product_name') if isinstance(x, dict) else np.nan
)

# ========== 5. PRODUCT COUNT BY CLUSTER + YEAR ==========
product_cluster_summary = (
    orders_exploded.groupby(['order_year', 'umap_cluster', 'product_name'])
    .size()
    .reset_index(name='count')
)

# ========== 6. CHURN RATE BY CLUSTER + CUSTOMER ROLE ==========
churn_summary = (
    orders_clustered
    .groupby(['umap_cluster', 'customer_role'])
    .agg(customers=('customer_id', 'nunique'),
         churned=('churned', 'sum'))
    .reset_index()
)

churn_summary['churn_rate'] = churn_summary['churned'] / churn_summary['customers']

# ========== 7. 📊 VISUALIZE CHURN RATE BY CLUSTER + ROLE ==========
plt.figure(figsize=(12, 6))
sns.barplot(data=churn_summary, x='umap_cluster', y='churn_rate', hue='customer_role', palette='Set2')
plt.title('Churn Rate by UMAP Cluster and Customer Role')
plt.xlabel('UMAP Cluster')
plt.ylabel('Churn Rate (%)')
plt.legend(title='Customer Role')
plt.tight_layout()
plt.show()

# ========== 8. 📊 PRODUCT TRENDS BY CLUSTER AND YEAR ==========
top_products = (
    product_cluster_summary
    .groupby('product_name')['count'].sum()
    .nlargest(10)
    .index
)

filtered_product_trends = product_cluster_summary[product_cluster_summary['product_name'].isin(top_products)]

plt.figure(figsize=(14, 7))
sns.lineplot(data=filtered_product_trends, x='order_year', y='count',
             hue='product_name', style='umap_cluster', palette='tab10')
plt.title('Top Product Trends by UMAP Cluster Over Time')
plt.xlabel('Year')
plt.ylabel('Order Count')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



