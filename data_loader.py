import pandas as pd
import numpy as np
import os
import hashlib
import ast
import re
from datetime import datetime, timedelta
import warnings

# Configure Numba for thread safety BEFORE importing UMAP
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '0'

# Terri's clustering dependencies
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import umap.umap_ as umap
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False

warnings.filterwarnings('ignore')

class DataLoader:
    """
    Data loading and preprocessing for ElleVet customer churn analysis
    """
    
    def __init__(self, data_path='data_fast/', fallback_path='data/'):
        self.data_path = data_path
        self.fallback_path = fallback_path
        self.use_fast_data = os.path.exists(f'{data_path}orders_redacted.pkl')
        
        if self.use_fast_data:
            print("⚡ Using reduced dataset for fast loading")
        else:
            print("🐌 Using full dataset (run python reduce_data.py for faster loading)")
            self.data_path = fallback_path
            
        self.customers_df = None
        self.orders_df = None
        self.orders_utm_df = None
        self.quizzes_df = None
        self.refunds_df = None
        self.subscriptions_df = None
        self.tickets_df = None
        
    def load_raw_data(self):
        """Load all raw data files (pickle if available, CSV as fallback)"""
        try:
            # Suppress DtypeWarnings for mixed types
            import warnings
            warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
            
            if self.use_fast_data:
                # Load from pickle files (much faster)
                self.customers_df = pd.read_pickle(f'{self.data_path}customers_redacted.pkl')
                self.orders_df = pd.read_pickle(f'{self.data_path}orders_redacted.pkl')
                self.orders_utm_df = pd.read_pickle(f'{self.data_path}orders_with_utm.pkl')
                self.quizzes_df = pd.read_pickle(f'{self.data_path}quizzes_redacted.pkl')
                self.refunds_df = pd.read_pickle(f'{self.data_path}refunds_affiliated.pkl')
                self.subscriptions_df = pd.read_pickle(f'{self.data_path}subscriptions_redacted.pkl')
                self.tickets_df = pd.read_pickle(f'{self.data_path}tickets_redacted.pkl')
                print("✅ All reduced data files loaded successfully (fast)")
            else:
                # Load from CSV files (slower, original data)
                self.customers_df = pd.read_csv(f'{self.data_path}customers_redacted.csv', low_memory=False)
                self.orders_df = pd.read_csv(f'{self.data_path}orders_redacted.csv', low_memory=False)
                self.orders_utm_df = pd.read_csv(f'{self.data_path}orders_with_utm.csv', low_memory=False)
                self.quizzes_df = pd.read_csv(f'{self.data_path}quizzes_redacted.csv', low_memory=False)
                self.refunds_df = pd.read_csv(f'{self.data_path}refunds_affiliated.csv', low_memory=False)
                self.subscriptions_df = pd.read_csv(f'{self.data_path}subscriptions_redacted.csv', low_memory=False)
                self.tickets_df = pd.read_csv(f'{self.data_path}tickets_redacted.csv', low_memory=False)
                print("✅ All CSV data files loaded successfully (slower)")
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def preprocess_dates(self):
        """Convert date columns to datetime format"""
        date_columns = {
            'orders_df': ['order_created_ts', 'order_created_gmt_ts', 'order_modified_ts',
                          'order_modified_gmt_ts', 'completed_date', 'paid_date'],
            'subscriptions_df': ['subscription_created_ts', 'subscription_last_modified_ts',
                                'subscription_next_payment_ts', 'subscription_cancelled_ts',
                                'subscription_ended_ts'],
            'tickets_df': ['created_at', 'updated_at'],
            'refunds_df': ['refund_date']
        }
        
        for df_name, cols in date_columns.items():
            df = getattr(self, df_name)
            for col in cols:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        if hasattr(df[col].dt, 'tz') and df[col].dt.tz is not None:
                            df[col] = df[col].dt.tz_localize(None)
                    except:
                        pass
    
    def classify_customers(self):
        """Classify customers as subscribers vs one-time customers"""
        def classify_customer_type(row):
            if row['is_subscription_start'] == True or row['is_subscription_renewal'] == True:
                return 'subscriber'
            else:
                return 'one_time_customer'
        
        # Add customer type to orders
        self.orders_df['customer_type'] = self.orders_df.apply(classify_customer_type, axis=1)
        
        # Get customer-level classification
        customer_types = self.orders_df.groupby('customer_id').agg({
            'customer_type': lambda x: 'subscriber' if 'subscriber' in x.values else 'one_time_customer',
            'order_id': 'count',
            'order_total': 'sum',
            'order_created_ts': ['min', 'max']
        }).reset_index()
        
        customer_types.columns = ['customer_id', 'customer_classification', 'total_orders', 
                                 'total_spent', 'first_order_date', 'last_order_date']
        
        return customer_types
    
    def extract_ticket_features(self):
        """Extract advanced ticket-based risk signals from Jason's approach"""
        def sha_email(email):
            return np.nan if pd.isna(email) else hashlib.sha256(email.strip().lower().encode()).hexdigest()
        
        # Create email hash for tickets
        self.tickets_df['email_hash'] = self.tickets_df['requester_email'].apply(sha_email)
        self.tickets_df['tags'] = self.tickets_df['tags'].fillna("").str.lower()
        self.tickets_df['tag_list'] = self.tickets_df['tags'].str.split(",")
        
        # Explode tags for analysis
        tix_exploded = self.tickets_df.explode('tag_list')
        tix_exploded['tag_list'] = tix_exploded['tag_list'].str.strip()
        
        # Create binary flags for problematic ticket patterns
        tix_exploded['has_cancel_tag'] = tix_exploded['tag_list'].str.contains(r"cancel", na=False)
        tix_exploded['has_urgent_tag'] = tix_exploded['tag_list'].str.contains(r"urgent", na=False)
        tix_exploded['mentions_dosing'] = tix_exploded['tag_list'].str.contains(r"dosing", na=False)
        tix_exploded['product_issue'] = tix_exploded['tag_list'].str.contains(
            r"product_|quality|broken|damage|leak", na=False
        )
        
        # Aggregate flags and counts by email
        ticket_flags = (
            tix_exploded.groupby('email_hash')[['has_cancel_tag', 'has_urgent_tag',
                                              'mentions_dosing', 'product_issue']]
            .max()
            .reset_index()
        )
        
        ticket_counts = (
            tix_exploded.groupby('email_hash').size()
            .reset_index(name='num_tickets')
        )
        
        # Combine features
        ticket_features = ticket_flags.merge(ticket_counts, on='email_hash', how='outer')
        
        # Fill NaN values and ensure integer types
        flag_cols = ['has_cancel_tag', 'has_urgent_tag', 'mentions_dosing', 'product_issue', 'num_tickets']
        for col in flag_cols:
            if col not in ticket_features.columns:
                ticket_features[col] = 0
            ticket_features[col] = (
                pd.to_numeric(ticket_features[col], errors='coerce')
                .fillna(0)
                .astype(int)
            )
        
        return ticket_features
    
    def extract_utm_risk_features(self):
        """Extract UTM-based risk signals from Jason's approach"""
        # Merge orders with UTM data
        orders_with_utm = self.orders_df[['order_id', 'customer_id']].merge(
            self.orders_utm_df[['order_id', 'utm_source']], on='order_id', how='left'
        )
        
        # Get first UTM source per customer
        first_utm = orders_with_utm.dropna(subset=['utm_source']).drop_duplicates('customer_id')
        first_utm['utm_source_clean'] = first_utm['utm_source'].str.lower()
        
        # Define problematic UTM sources
        risky_utm_patterns = ['facebook', 'fb', 'affiliate', 'coupon', 'discount']
        first_utm['utm_risk_flag'] = first_utm['utm_source_clean'].apply(
            lambda x: int(any(pattern in str(x) for pattern in risky_utm_patterns))
        )
        
        return first_utm[['customer_id', 'utm_risk_flag']]
    
    def extract_product_features(self):
        """Extract product form classification from Jason's approach"""
        def extract_product_names(line_items):
            try:
                items = ast.literal_eval(line_items) if isinstance(line_items, str) else []
                return [item.get('product_name', '') for item in items if isinstance(item, dict)]
            except Exception:
                return []
        
        def classify_product_form(product_name):
            if not isinstance(product_name, str):
                return 'Other'
            name_lower = product_name.lower()
            if 'chew' in name_lower:
                return 'Chew'
            elif 'softgel' in name_lower or 'gel' in name_lower:
                return 'Softgel'
            elif 'oil' in name_lower:
                return 'Oil'
            elif 'capsule' in name_lower:
                return 'Capsule'
            return 'Other'
        
        # Extract product names from line items
        self.orders_df['product_names'] = self.orders_df['line_items'].apply(extract_product_names)
        orders_exploded = self.orders_df.explode('product_names')
        orders_exploded['product_form'] = orders_exploded['product_names'].apply(classify_product_form)
        
        # Get first product form per customer
        first_product_form = (
            orders_exploded.sort_values('order_created_ts')
            .drop_duplicates('customer_id')[['customer_id', 'product_form']]
        )
        
        return first_product_form
    
    def extract_quiz_features(self):
        """Extract quiz-based risk signals from Jason's approach"""
        quiz_flags = self.quizzes_df[['email', '16._pet_type', '24._situational_stress']].copy()
        quiz_flags = quiz_flags.rename(columns={'email': 'email_hash'})
        
        # Create cat ownership flag
        quiz_flags['cat_flag'] = (
            quiz_flags['16._pet_type'].fillna('')
            .str.lower()
            .str.contains('cat')
            .astype(int)
        )
        
        # Create stress situation flag
        quiz_flags['stress_flag'] = (
            quiz_flags['24._situational_stress'].fillna('')
            .str.lower()
            .str.contains('stress')
            .astype(int)
        )
        
        return quiz_flags[['email_hash', 'cat_flag', 'stress_flag']]
    
    def extract_refund_features(self):
        """Extract refund-based risk signals from Jason's approach"""
        # Merge refunds with orders to get customer_id
        refunds_with_customers = self.refunds_df.merge(
            self.orders_df[['order_id', 'customer_id']], on='order_id', how='left'
        )
        
        # Create binary refund flag per customer
        refund_flags = refunds_with_customers[['customer_id']].drop_duplicates()
        refund_flags['has_refund'] = 1
        
        return refund_flags
    
    def extract_john_discount_features(self):
        """Extract John's discount usage patterns"""
        # Detect discount usage (similar to John's approach)
        discount_features = self.orders_df.copy()
        discount_features['discount_used'] = (discount_features['cart_discount'].fillna(0).astype(float) > 0)
        
        # Aggregate by customer
        customer_discount = discount_features.groupby('customer_id').agg({
            'discount_used': 'max',  # Ever used discount
            'cart_discount': ['sum', 'count']  # Total discounts and frequency
        }).reset_index()
        
        # Flatten column names
        customer_discount.columns = ['customer_id', 'ever_used_discount', 'total_discount_amount', 'discount_frequency']
        customer_discount['ever_used_discount'] = customer_discount['ever_used_discount'].astype(int)
        
        return customer_discount
    
    def extract_john_support_patterns(self):
        """Extract John's support issue patterns from ticket tags"""
        try:
            if self.tickets_df is None or len(self.tickets_df) == 0:
                return pd.DataFrame(columns=['email_hash', 'escalated_flag', 'site_flag'])
            
            # Clean ticket data similar to John's approach
            tickets_clean = self.tickets_df.copy()
            tickets_clean['email_hash'] = tickets_clean['requester_email'].fillna('').str.lower().str.strip()
            tickets_clean['tags'] = tickets_clean['tags'].fillna('').str.lower()
            
            # Aggregate tags by customer email
            tag_aggregated = tickets_clean.groupby('email_hash')['tags'].apply(lambda x: ','.join(x)).reset_index()
            
            # Extract John's patterns
            tag_aggregated['escalated_flag'] = tag_aggregated['tags'].str.contains('escalated', na=False).astype(int)
            tag_aggregated['site_flag'] = tag_aggregated['tags'].str.contains('site|website|checkout|page', na=False).astype(int)
            
            return tag_aggregated[['email_hash', 'escalated_flag', 'site_flag']]
            
        except Exception as e:
            print(f"⚠️  John's support pattern extraction failed: {e}")
            return pd.DataFrame(columns=['email_hash', 'escalated_flag', 'site_flag'])
    
    def extract_john_spend_tiers(self, customer_features):
        """Extract John's spend tier analysis"""
        try:
            # Create spend tiers using John's quantile approach
            customer_features['spend_tier'] = pd.qcut(
                customer_features['total_spent'], 
                q=3, 
                labels=['Low', 'Mid', 'High'],
                duplicates='drop'  # Handle edge cases
            ).astype(str)
            
            # Create binary flags like John's approach
            customer_features['spend_tier_low'] = (customer_features['spend_tier'] == 'Low').astype(int)
            customer_features['spend_tier_mid'] = (customer_features['spend_tier'] == 'Mid').astype(int)
            customer_features['spend_tier_high'] = (customer_features['spend_tier'] == 'High').astype(int)
            
            return customer_features
            
        except Exception as e:
            print(f"⚠️  John's spend tier creation failed: {e}")
            # Create default tiers
            customer_features['spend_tier'] = 'Mid'
            customer_features['spend_tier_low'] = 0
            customer_features['spend_tier_mid'] = 1
            customer_features['spend_tier_high'] = 0
            return customer_features
    
    def compute_jason_risk_score(self, customer_features):
        """Compute Jason's heuristic risk score (0-8 scale)"""
        # Initialize score
        customer_features['jason_risk_score'] = (
            1 * (customer_features.get('has_cancel_tag', 0) == 1) +
            1 * (customer_features.get('product_issue', 0) == 1) +
            1 * (customer_features.get('has_urgent_tag', 0) == 1) +
            1 * (customer_features.get('num_tickets', 0) >= 2) +
            1 * (customer_features.get('has_refund', 0) == 1) +
            1 * (customer_features.get('utm_risk_flag', 0) == 1) +
            1 * (customer_features.get('avg_order_value', 0) < 40) +
            1 * (customer_features.get('days_since_last_order', 0) > 120)
        )
        
        return customer_features
    
    def compute_customer_clusters(self, customer_features):
        """Compute UMAP and PCA clusters based on Terri's approach"""
        if not CLUSTERING_AVAILABLE:
            print("⚠️  Clustering packages not available, skipping cluster analysis")
            customer_features['umap_cluster'] = 0
            customer_features['pca_cluster'] = 0
            return customer_features
        
        try:
            print("🔬 Computing customer clusters (Terri's approach)...")
            
            # Select numerical features for clustering
            cluster_features = [
                'total_orders', 'total_spent', 'days_since_first_order', 'days_since_last_order',
                'customer_lifetime_days', 'avg_order_value', 'order_value_std', 'min_order_value',
                'max_order_value', 'avg_shipping', 'total_discounts', 'total_tax',
                'refund_count', 'total_refunded', 'num_tickets', 'jason_risk_score'
            ]
            
            # Filter features that actually exist in the data
            available_features = [f for f in cluster_features if f in customer_features.columns]
            
            if len(available_features) < 4:
                print("⚠️  Insufficient features for clustering")
                customer_features['umap_cluster'] = 0
                customer_features['pca_cluster'] = 0
                return customer_features
            
            # Prepare clustering data
            clustering_data = customer_features[available_features].fillna(0)
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(clustering_data)
            
            # PCA dimensionality reduction
            pca = PCA(n_components=min(2, len(available_features)), random_state=42)
            pca_components = pca.fit_transform(scaled_features)
            
            # UMAP dimensionality reduction with thread safety and fallback
            try:
                reducer = umap.UMAP(
                    n_components=2, 
                    random_state=42, 
                    n_neighbors=min(15, len(customer_features)//2),
                    n_jobs=1,  # Single thread to avoid Numba threading issues
                    low_memory=True,
                    verbose=False,  # Reduce output noise
                    transform_seed=42  # Additional reproducibility
                )
                umap_components = reducer.fit_transform(scaled_features)
                umap_success = True
                print("✅ UMAP clustering successful")
            except Exception as e:
                print(f"⚠️  UMAP clustering failed: {e}")
                print("🔄 Using PCA components for both clustering methods")
                umap_components = pca_components
                umap_success = False
            
            # Determine optimal number of clusters
            best_k = self._find_optimal_clusters(pca_components, max_k=min(8, len(customer_features)//10))
            
            # K-means clustering on PCA components
            kmeans_pca = KMeans(n_clusters=best_k, random_state=42)
            customer_features['pca_cluster'] = kmeans_pca.fit_predict(pca_components)
            
            # K-means clustering on UMAP components (or PCA fallback)
            kmeans_umap = KMeans(n_clusters=best_k, random_state=42)
            customer_features['umap_cluster'] = kmeans_umap.fit_predict(umap_components)
            
            # Store clustering components for analysis
            customer_features['pca_1'] = pca_components[:, 0]
            customer_features['pca_2'] = pca_components[:, 1] if pca_components.shape[1] > 1 else 0
            customer_features['umap_1'] = umap_components[:, 0]
            customer_features['umap_2'] = umap_components[:, 1]
            
            print(f"✅ Customer clustering complete: {best_k} clusters identified")
            print(f"   PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
            
            # Print cluster distribution
            pca_dist = customer_features['pca_cluster'].value_counts().sort_index()
            umap_dist = customer_features['umap_cluster'].value_counts().sort_index()
            print(f"   PCA cluster sizes: {dict(pca_dist)}")
            print(f"   UMAP cluster sizes: {dict(umap_dist)}")
            
            return customer_features
            
        except Exception as e:
            print(f"⚠️  Clustering failed: {e}")
            customer_features['umap_cluster'] = 0
            customer_features['pca_cluster'] = 0
            customer_features['pca_1'] = 0
            customer_features['pca_2'] = 0
            customer_features['umap_1'] = 0
            customer_features['umap_2'] = 0
            return customer_features
    
    def _find_optimal_clusters(self, data, max_k=8):
        """Find optimal number of clusters using silhouette score"""
        if len(data) < 10:
            return 2
        
        best_score = -1
        best_k = 2
        
        for k in range(2, min(max_k + 1, len(data)//2)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(data)
                score = silhouette_score(data, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
        
        return best_k
    
    def analyze_cluster_characteristics(self, customer_features):
        """Analyze characteristics of each cluster (Terri's approach)"""
        if 'umap_cluster' not in customer_features.columns:
            return None
        
        try:
            cluster_analysis = {}
            
            for cluster_col in ['umap_cluster', 'pca_cluster']:
                if cluster_col not in customer_features.columns:
                    continue
                
                # Build aggregation dict based on available columns
                agg_dict = {
                    'total_spent': ['mean', 'median', 'std'],
                    'total_orders': ['mean', 'median'],
                    'days_since_last_order': ['mean', 'median'],
                    'jason_risk_score': ['mean', 'median'],
                    'customer_id': 'count'
                }
                
                # Only add churn_probability if it exists
                if 'churn_probability' in customer_features.columns:
                    agg_dict['churn_probability'] = ['mean']
                
                cluster_stats = customer_features.groupby(cluster_col).agg(agg_dict).round(2)
                
                cluster_analysis[cluster_col] = cluster_stats
            
            return cluster_analysis
            
        except Exception as e:
            print(f"⚠️  Cluster analysis failed: {e}")
            return None
    
    def analyze_product_preferences(self, customer_features):
        """Analyze product preferences by risk level and customer cluster"""
        try:
            print("🔬 Analyzing product preferences by risk and cluster...")
            
            # Create risk level categories
            def categorize_risk(churn_prob):
                if churn_prob > 0.7:
                    return 'High Risk'
                elif churn_prob > 0.4:
                    return 'Medium Risk'
                else:
                    return 'Low Risk'
            
            # Add risk categories if churn probability exists
            if 'churn_probability' in customer_features.columns:
                customer_features['risk_category'] = customer_features['churn_probability'].apply(categorize_risk)
            else:
                customer_features['risk_category'] = 'Unknown'
            
            # Analyze product form preferences by risk level
            if 'product_form' in customer_features.columns:
                risk_product_analysis = customer_features.groupby(['risk_category', 'product_form']).agg({
                    'customer_id': 'count',
                    'total_spent': 'mean',
                    'jason_risk_score': 'mean'
                }).reset_index()
                
                risk_product_analysis.columns = ['risk_category', 'product_form', 'customer_count', 'avg_spent', 'avg_jason_score']
                
                # Analyze by cluster if available
                cluster_product_analysis = None
                if 'umap_cluster' in customer_features.columns:
                    cluster_product_analysis = customer_features.groupby(['umap_cluster', 'product_form']).agg({
                        'customer_id': 'count',
                        'total_spent': 'mean',
                        'churn_probability': 'mean' if 'churn_probability' in customer_features.columns else 'count'
                    }).reset_index()
                    
                    cluster_product_analysis.columns = ['cluster', 'product_form', 'customer_count', 'avg_spent', 'avg_churn']
                
                return {
                    'risk_product_analysis': risk_product_analysis,
                    'cluster_product_analysis': cluster_product_analysis
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️  Product preference analysis failed: {e}")
            return None
    
    def create_customer_features(self, customer_types):
        """Create features for churn prediction"""
        
        # Focus on one-time customers only
        one_time_customers = customer_types[
            customer_types['customer_classification'] == 'one_time_customer'
        ].copy()
        
        # Current date for calculations
        current_date = datetime.now()
        
        # Basic temporal features
        one_time_customers['days_since_first_order'] = (
            current_date - one_time_customers['first_order_date']
        ).dt.days
        
        one_time_customers['days_since_last_order'] = (
            current_date - one_time_customers['last_order_date']
        ).dt.days
        
        one_time_customers['customer_lifetime_days'] = (
            one_time_customers['last_order_date'] - one_time_customers['first_order_date']
        ).dt.days
        
        # Handle single-order customers
        one_time_customers['customer_lifetime_days'] = one_time_customers['customer_lifetime_days'].fillna(0)
        
        # Merge with customer demographic data (email_hash already exists)
        customer_features = one_time_customers.merge(
            self.customers_df[['customer_id', 'email_hash', 'shipping_city', 'shipping_state', 'shipping_postcode']],
            on='customer_id',
            how='left'
        )
        
        # Order behavior features
        customer_orders = self.orders_df[
            self.orders_df['customer_id'].isin(customer_features['customer_id'])
        ].groupby('customer_id').agg({
            'order_total': ['mean', 'std', 'min', 'max'],
            'order_shipping': 'mean',
            'cart_discount': 'sum',
            'order_tax': 'sum',
            'created_via': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown',
            'payment_method': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
        }).reset_index()
        
        # Flatten column names
        customer_orders.columns = ['customer_id', 'avg_order_value', 'order_value_std', 
                                  'min_order_value', 'max_order_value', 'avg_shipping',
                                  'total_discounts', 'total_tax', 'primary_channel', 'primary_payment']
        
        # Fill NaN values for single-order customers
        customer_orders['order_value_std'] = customer_orders['order_value_std'].fillna(0)
        
        # Merge order features
        customer_features = customer_features.merge(customer_orders, on='customer_id', how='left')
        
        # UTM/Marketing features (existing approach)
        utm_features = self.orders_df.merge(
            self.orders_utm_df[['order_id', 'utm_source', 'utm_medium', 'utm_campaign']],
            on='order_id',
            how='left'
        ).groupby('customer_id').agg({
            'utm_source': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown',
            'utm_medium': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
        }).reset_index()
        
        utm_features.columns = ['customer_id', 'primary_utm_source', 'primary_utm_medium']
        customer_features = customer_features.merge(utm_features, on='customer_id', how='left')
        
        # === JASON'S ADVANCED FEATURES INTEGRATION ===
        print("🔬 Extracting Jason's advanced features...")
        
        # Extract ticket-based risk signals
        try:
            ticket_features = self.extract_ticket_features()
            customer_features = customer_features.merge(ticket_features, on='email_hash', how='left')
            print("✅ Ticket features extracted")
        except Exception as e:
            print(f"⚠️  Ticket features failed: {e}")
            # Fill with defaults
            for col in ['has_cancel_tag', 'has_urgent_tag', 'mentions_dosing', 'product_issue', 'num_tickets']:
                customer_features[col] = 0
        
        # === JOHN'S FEATURE INTEGRATION ===
        print("🔬 Extracting John's behavioral features...")
        
        # Extract John's discount patterns
        try:
            discount_features = self.extract_john_discount_features()
            customer_features = customer_features.merge(discount_features, on='customer_id', how='left')
            for col in ['ever_used_discount', 'total_discount_amount', 'discount_frequency']:
                customer_features[col] = customer_features[col].fillna(0).astype(int if col != 'total_discount_amount' else float)
            print("✅ John's discount features extracted")
        except Exception as e:
            print(f"⚠️  John's discount features failed: {e}")
            customer_features['ever_used_discount'] = 0
            customer_features['total_discount_amount'] = 0.0
            customer_features['discount_frequency'] = 0
        
        # Extract John's support patterns
        try:
            support_patterns = self.extract_john_support_patterns()
            customer_features = customer_features.merge(support_patterns, on='email_hash', how='left')
            for col in ['escalated_flag', 'site_flag']:
                customer_features[col] = customer_features[col].fillna(0).astype(int)
            print("✅ John's support patterns extracted")
        except Exception as e:
            print(f"⚠️  John's support patterns failed: {e}")
            customer_features['escalated_flag'] = 0
            customer_features['site_flag'] = 0
        
        # Extract UTM risk flags
        try:
            utm_risk_features = self.extract_utm_risk_features()
            customer_features = customer_features.merge(utm_risk_features, on='customer_id', how='left')
            customer_features['utm_risk_flag'] = customer_features['utm_risk_flag'].fillna(0).astype(int)
            print("✅ UTM risk features extracted")
        except Exception as e:
            print(f"⚠️  UTM risk features failed: {e}")
            customer_features['utm_risk_flag'] = 0
        
        # Extract product form features
        try:
            product_features = self.extract_product_features()
            customer_features = customer_features.merge(product_features, on='customer_id', how='left')
            customer_features['product_form'] = customer_features['product_form'].fillna('Other')
            print("✅ Product form features extracted")
        except Exception as e:
            print(f"⚠️  Product form features failed: {e}")
            customer_features['product_form'] = 'Other'
        
        # Extract quiz-based features
        try:
            quiz_features = self.extract_quiz_features()
            customer_features = customer_features.merge(quiz_features, on='email_hash', how='left')
            customer_features['cat_flag'] = customer_features['cat_flag'].fillna(0).astype(int)
            customer_features['stress_flag'] = customer_features['stress_flag'].fillna(0).astype(int)
            print("✅ Quiz features extracted")
        except Exception as e:
            print(f"⚠️  Quiz features failed: {e}")
            customer_features['cat_flag'] = 0
            customer_features['stress_flag'] = 0
        
        # Extract refund features
        try:
            refund_features = self.extract_refund_features()
            customer_features = customer_features.merge(refund_features, on='customer_id', how='left')
            customer_features['has_refund'] = customer_features['has_refund'].fillna(0).astype(int)
            print("✅ Refund features extracted")
        except Exception as e:
            print(f"⚠️  Refund features failed: {e}")
            customer_features['has_refund'] = 0
        
        # Ensure all Jason's features have default values
        jason_feature_defaults = {
            'has_cancel_tag': 0, 'has_urgent_tag': 0, 'mentions_dosing': 0, 
            'product_issue': 0, 'num_tickets': 0, 'utm_risk_flag': 0,
            'cat_flag': 0, 'stress_flag': 0, 'has_refund': 0
        }
        
        for feature, default_value in jason_feature_defaults.items():
            if feature not in customer_features.columns:
                customer_features[feature] = default_value
            customer_features[feature] = customer_features[feature].fillna(default_value).astype(int)
        
        # Legacy ticket count for backward compatibility
        customer_features['ticket_count'] = customer_features['num_tickets']
        
        # Enhanced refund features (keeping both old and new approach)
        refund_amount_features = self.orders_df.merge(
            self.refunds_df[['order_id', 'refund_amount']],
            on='order_id',
            how='left'
        ).groupby('customer_id').agg({
            'refund_amount': ['count', 'sum']
        }).reset_index()
        
        refund_amount_features.columns = ['customer_id', 'refund_count', 'total_refunded']
        customer_features = customer_features.merge(refund_amount_features, on='customer_id', how='left')
        customer_features['refund_count'] = customer_features['refund_count'].fillna(0)
        customer_features['total_refunded'] = customer_features['total_refunded'].fillna(0)
        
        # Compute Jason's heuristic risk score (0-8)
        customer_features = self.compute_jason_risk_score(customer_features)
        print("✅ Jason's risk score computed")
        
        # Add John's spend tier analysis (after all other features)
        try:
            customer_features = self.extract_john_spend_tiers(customer_features)
            print("✅ John's spend tiers created")
        except Exception as e:
            print(f"⚠️  John's spend tiers failed: {e}")
        
        # Create John's interaction features (after all base features exist)
        try:
            # Discount + Refund interaction (high-risk pattern)
            customer_features['discount_refund_interaction'] = (
                (customer_features.get('ever_used_discount', 0) == 1) & 
                (customer_features.get('has_refund', 0) == 1)
            ).astype(int)
            
            # Site issues + Refund interaction (website problems leading to refunds)
            customer_features['site_refund_interaction'] = (
                (customer_features.get('site_flag', 0) == 1) & 
                (customer_features.get('has_refund', 0) == 1)
            ).astype(int)
            
            # Escalated + Discount interaction (price-sensitive complaints)
            customer_features['escalated_discount_interaction'] = (
                (customer_features.get('escalated_flag', 0) == 1) & 
                (customer_features.get('ever_used_discount', 0) == 1)
            ).astype(int)
            
            print("✅ John's interaction features created")
        except Exception as e:
            print(f"⚠️  John's interaction features failed: {e}")
            customer_features['discount_refund_interaction'] = 0
            customer_features['site_refund_interaction'] = 0
            customer_features['escalated_discount_interaction'] = 0
        
        # === TERRI'S CLUSTERING FEATURES INTEGRATION ===
        print("🔬 Computing Terri's customer clusters...")
        customer_features = self.compute_customer_clusters(customer_features)
        
        # Analyze cluster characteristics
        cluster_analysis = self.analyze_cluster_characteristics(customer_features)
        if cluster_analysis:
            print("✅ Cluster analysis complete")
        else:
            print("⚠️  Cluster analysis skipped")
        
        # Create churn labels based on BUSINESS-RELEVANT rules for actionable timeframe
        # Fixed churn threshold for business consistency: 180 days is clearly churned behavior
        churn_threshold = 180  # Fixed threshold - customers who haven't ordered in 6+ months
        
        customer_features['is_churned'] = (customer_features['days_since_last_order'] >= churn_threshold).astype(int)
        
        # Ensure we have both classes for model training
        churn_rate = customer_features['is_churned'].mean()
        if churn_rate > 0.95 or churn_rate < 0.05:
            # If still imbalanced, use 150 days as backup threshold
            churn_threshold = 150
            customer_features['is_churned'] = (customer_features['days_since_last_order'] >= churn_threshold).astype(int)
            churn_rate = customer_features['is_churned'].mean()
            
            # Final fallback to 120 days if still imbalanced
            if churn_rate > 0.95 or churn_rate < 0.05:
                churn_threshold = 120
                customer_features['is_churned'] = (customer_features['days_since_last_order'] >= churn_threshold).astype(int)
        
        # Risk categories - aligned with churn threshold and actionable timeframes
        def categorize_risk(days_since_last):
            if days_since_last >= churn_threshold:  # Use actual churn threshold
                return 'churned'
            elif days_since_last >= 120:  # 120+ days - very high risk
                return 'very_high_risk'
            elif days_since_last >= 90:   # 90-120 days - high risk
                return 'high_risk'
            elif days_since_last >= 60:   # 60-90 days - medium risk
                return 'medium_risk'
            elif days_since_last >= 30:   # 30-60 days - low risk
                return 'low_risk'
            else:  # 0-30 days - very low risk
                return 'very_low_risk'
        
        customer_features['risk_category'] = customer_features['days_since_last_order'].apply(categorize_risk)
        
        print(f"🎯 Adaptive churn definition for balanced modeling:")
        print(f"   Churn threshold: {churn_threshold:.0f} days")
        print(f"   Churn rate: {customer_features['is_churned'].mean():.1%}")
        print(f"   Churned: {customer_features['is_churned'].sum():,} customers")
        print(f"   Retained: {(~customer_features['is_churned'].astype(bool)).sum():,} customers")
        
        return customer_features
    
    def load_and_process_data(self):
        """Main method to load and process all data"""
        print("🔄 Loading ElleVet customer data...")
        
        # Load raw data
        if not self.load_raw_data():
            raise Exception("Failed to load raw data")
        
        # Preprocess dates
        self.preprocess_dates()
        print("✅ Date preprocessing complete")
        
        # Classify customers
        customer_types = self.classify_customers()
        print(f"✅ Customer classification complete: {len(customer_types)} customers")
        
        # Create features
        customer_features = self.create_customer_features(customer_types)
        print(f"✅ Feature engineering complete: {customer_features.shape[1]} features")
        
        # Filter to RECENT customers only (last 18 months for business relevance)
        # Focus on customers who are still potentially active, not long-lost customers
        customers_before_filter = len(customer_features)
        recent_cutoff = datetime.now() - timedelta(days=540)  # 18 months
        customer_features = customer_features[
            customer_features['last_order_date'] >= recent_cutoff
        ]
        
        print(f"🎯 Filtering to recent customers (purchased within last 18 months)")
        print(f"   Before filter: {customers_before_filter} customers")
        print(f"   After filter: {len(customer_features)} customers")
        
        print(f"✅ Data processing complete: {len(customer_features)} customers in final dataset")
        print(f"   Churn rate: {customer_features['is_churned'].mean():.1%}")
        print(f"   Date range: {customer_features['first_order_date'].min().strftime('%Y-%m')} to {customer_features['first_order_date'].max().strftime('%Y-%m')}")
        
        return customer_features
    
    def get_data_summary(self):
        """Get summary statistics of the loaded data"""
        if self.orders_df is None:
            return "No data loaded"
        
        summary = {
            'total_customers': self.customers_df['customer_id'].nunique(),
            'total_orders': len(self.orders_df),
            'date_range': f"{self.orders_df['order_created_ts'].min()} to {self.orders_df['order_created_ts'].max()}",
            'total_revenue': self.orders_df['order_total'].sum()
        }
        
        return summary