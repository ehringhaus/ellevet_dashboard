#!/usr/bin/env python3
"""
Quick fix for timezone datetime comparison error
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def fix_and_reduce_data():
    """
    Fixed version that handles timezone issues properly
    """
    
    print("🔧 ElleVet Data Reducer - Fixed Version")
    print("=" * 45)
    
    # Create output directory
    output_dir = Path('data_fast')
    output_dir.mkdir(exist_ok=True)
    
    # Date filter - only keep last 18 months of data  
    cutoff_date = datetime.now() - timedelta(days=540)
    print(f"📅 Filtering to data after: {cutoff_date.strftime('%Y-%m-%d')}")
    
    try:
        # 1. ORDERS - Process with timezone handling
        print("\n📊 Processing orders_redacted.csv...")
        orders_df = pd.read_csv('data/orders_redacted.csv', low_memory=False)
        original_size = len(orders_df)
        
        # Convert and normalize datetime column
        orders_df['order_created_ts'] = pd.to_datetime(orders_df['order_created_ts'], errors='coerce')
        
        # Handle timezone issues
        if orders_df['order_created_ts'].dt.tz is not None:
            print("   🕐 Normalizing timezone-aware dates...")
            orders_df['order_created_ts'] = orders_df['order_created_ts'].dt.tz_localize(None)
        
        # Filter to recent orders (handle NaT values)
        valid_dates = orders_df['order_created_ts'].notna()
        recent_dates = orders_df['order_created_ts'] >= cutoff_date
        orders_df = orders_df[valid_dates & recent_dates]
        
        # Filter to non-subscriber orders only (one-time customers)
        orders_df = orders_df[
            (orders_df['is_subscription_renewal'] != True) & 
            (orders_df['is_subscription_start'] != True)
        ]
        
        print(f"   Original: {original_size:,} rows")
        print(f"   Filtered: {len(orders_df):,} rows ({len(orders_df)/original_size*100:.1f}%)")
        
        # Save as pickle
        orders_df.to_pickle(f'{output_dir}/orders_redacted.pkl')
        
        # Get relevant IDs for filtering other files
        relevant_order_ids = set(orders_df['order_id'])
        relevant_customer_ids = set(orders_df['customer_id'])
        
        print(f"   📋 Found {len(relevant_customer_ids):,} relevant customers")
        print(f"   📋 Found {len(relevant_order_ids):,} relevant orders")
        
    except Exception as e:
        print(f"❌ Error processing orders: {e}")
        return
    
    # Process other files with error handling
    files_to_process = [
        ('customers_redacted.csv', 'customer_id', relevant_customer_ids, None),
        ('orders_with_utm.csv', 'order_id', relevant_order_ids, None), 
        ('refunds_affiliated.csv', 'order_id', relevant_order_ids, None),
        ('tickets_redacted.csv', None, None, 'created_at'),
        ('quizzes_redacted.csv', None, None, 'date'),
        ('subscriptions_redacted.csv', None, None, None)  # Keep all
    ]
    
    total_original_mb = 0
    total_new_mb = 0
    
    for csv_file, filter_col, filter_ids, date_col in files_to_process:
        try:
            print(f"\n📊 Processing {csv_file}...")
            
            if not os.path.exists(f'data/{csv_file}'):
                print(f"   ⚠️  File not found, skipping")
                continue
                
            df = pd.read_csv(f'data/{csv_file}', low_memory=False)
            original_size = len(df)
            
            # Filter by ID if specified
            if filter_col and filter_ids:
                df = df[df[filter_col].isin(filter_ids)]
            
            # Filter by date if specified
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Handle timezone issues
                if df[date_col].dt.tz is not None:
                    df[date_col] = df[date_col].dt.tz_localize(None)
                
                # Filter to recent data only
                valid_dates = df[date_col].notna()
                recent_dates = df[date_col] >= cutoff_date
                df = df[valid_dates & recent_dates]
            
            print(f"   Original: {original_size:,} rows")
            print(f"   Filtered: {len(df):,} rows ({len(df)/original_size*100:.1f}%)")
            
            # Save as pickle
            pkl_name = csv_file.replace('.csv', '.pkl')
            df.to_pickle(f'{output_dir}/{pkl_name}')
            
            # Calculate size comparison
            original_size_mb = os.path.getsize(f'data/{csv_file}') / (1024 * 1024)
            new_size_mb = os.path.getsize(f'{output_dir}/{pkl_name}') / (1024 * 1024)
            
            total_original_mb += original_size_mb
            total_new_mb += new_size_mb
            
            reduction_pct = (1 - new_size_mb / original_size_mb) * 100
            print(f"   Size: {original_size_mb:.1f}MB → {new_size_mb:.1f}MB ({reduction_pct:.1f}% smaller)")
            
        except Exception as e:
            print(f"   ❌ Error processing {csv_file}: {e}")
            continue
    
    # Final summary
    if total_original_mb > 0:
        total_reduction_pct = (1 - total_new_mb / total_original_mb) * 100
        speedup_factor = total_original_mb / total_new_mb
        
        print(f"\n🎉 Final Results:")
        print(f"   Original total: {total_original_mb:.1f} MB")
        print(f"   Reduced total: {total_new_mb:.1f} MB")
        print(f"   Total savings: {total_reduction_pct:.1f}% smaller")
        print(f"   Expected speedup: {speedup_factor:.1f}x faster")
        
        print(f"\n✅ Success! Reduced files saved to: {output_dir}/")
        print(f"🚀 Your dashboard will now load much faster!")
        
        return True
    else:
        print("\n❌ No files processed successfully")
        return False

if __name__ == '__main__':
    fix_and_reduce_data()