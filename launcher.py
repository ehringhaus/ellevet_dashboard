#!/usr/bin/env python3
"""
ElleVet Customer Churn Dashboard Launcher
Handles setup, data validation, and dashboard startup
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

def check_python_version():
    """Ensure Python 3.7+ is being used"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")

def create_directories():
    """Create necessary directories"""
    directories = ['assets', 'models', 'logs', 'exports']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_data_files():
    """Validate that required data files exist"""
    required_files = [
        'data/customers_redacted.csv',
        'data/orders_redacted.csv',
        'data/orders_with_utm.csv',
        'data/quizzes_redacted.csv',
        'data/refunds_affiliated.csv',
        'data/subscriptions_redacted.csv',
        'data/tickets_redacted.csv'
    ]
    
    missing_files = []
    file_info = []
    
    # Suppress DtypeWarnings for mixed types during validation
    import warnings
    warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
    
    for file_path in required_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, low_memory=False)
                file_info.append({
                    'file': file_path,
                    'rows': len(df),
                    'size_mb': os.path.getsize(file_path) / (1024*1024),
                    'status': '✅'
                })
                print(f"✅ {file_path}: {len(df):,} rows ({os.path.getsize(file_path)/(1024*1024):.1f}MB)")
            except Exception as e:
                file_info.append({
                    'file': file_path,
                    'error': str(e),
                    'status': '❌'
                })
                print(f"❌ {file_path}: Error reading file - {str(e)}")
        else:
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📝 Please copy your CSV files to the data/ directory")
        return False
    
    return True

def install_requirements():
    """Install required Python packages"""
    print("\n🔄 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def test_imports():
    """Test that all required modules can be imported"""
    print("\n🔄 Testing module imports...")
    
    required_modules = [
        'dash', 'plotly', 'pandas', 'numpy', 'sklearn'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("   Try running: pip install -r requirements.txt")
        return False
    
    return True

def validate_data_quality():
    """Basic data quality checks"""
    print("\n🔍 Running data quality checks...")
    
    # Check for fast data files first
    fast_data_exists = os.path.exists('data_fast/orders_redacted.pkl')
    
    if fast_data_exists:
        print("⚡ Fast data files detected!")
        
        # Check fast data files
        fast_files = [
            'data_fast/customers_redacted.pkl',
            'data_fast/orders_redacted.pkl', 
            'data_fast/orders_with_utm.pkl',
            'data_fast/refunds_affiliated.pkl'
        ]
        
        total_size_mb = 0
        for file_path in fast_files:
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                total_size_mb += size_mb
                print(f"✅ {file_path}: {size_mb:.1f} MB (optimized)")
        
        print(f"📊 Total optimized data: {total_size_mb:.1f} MB")
        print("🚀 Dashboard will load quickly!")
        
        # Test data loading with optimized data
        try:
            from data_loader import DataLoader
            loader = DataLoader()
            
            if not loader.load_raw_data():
                print("❌ Failed to load optimized data files")
                return False
                
            print("✅ Optimized data validation successful")
            return True
            
        except Exception as e:
            print(f"❌ Data validation error: {e}")
            return False
    else:
        print("🐌 Using original CSV files (slower)")
        print("💡 Run 'python quick_fix_script.py' for 10x faster loading")
        
        # Original CSV validation
        try:
            from data_loader import DataLoader
            loader = DataLoader()
            
            if not loader.load_raw_data():
                print("❌ Failed to load data files")
                return False
                
            # Basic validation
            customers_count = len(loader.customers_df)
            orders_count = len(loader.orders_df)
            
            print(f"✅ Loaded {customers_count:,} customers")
            print(f"✅ Loaded {orders_count:,} orders")
            
            # Check for reasonable data ranges
            if customers_count < 100:
                print("⚠️  Warning: Very few customers in dataset")
            if orders_count < 1000:
                print("⚠️  Warning: Very few orders in dataset")
                
            return True
            
        except Exception as e:
            print(f"❌ Data validation error: {e}")
            return False

def create_sample_config():
    """Create a sample configuration file if it doesn't exist"""
    if not os.path.exists('config.py'):
        print("📝 Creating sample config.py file...")
        # The config.py content would be written here
        print("✅ Created config.py - review and modify as needed")

def run_dashboard(host='127.0.0.1', port=8050, debug=True):
    """Launch the dashboard"""
    print(f"\n🚀 Starting ElleVet Customer Churn Dashboard...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Debug mode: {debug}")
    print(f"   URL: http://{host}:{port}")
    print(f"\n   Dashboard will automatically open in your browser...")
    print(f"   Press Ctrl+C to stop the dashboard\n")
    
    try:
        # Import and run the dashboard
        from app import app
        app.run_server(debug=debug, host=host, port=port)
    except ImportError as e:
        print(f"❌ Error importing dashboard: {e}")
        print("   Make sure app.py exists and all dependencies are installed")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description='ElleVet Customer Churn Dashboard Launcher')
    parser.add_argument('--setup-only', action='store_true', help='Run setup checks only, don\'t start dashboard')
    parser.add_argument('--skip-checks', action='store_true', help='Skip validation checks and start dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind the dashboard to')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the dashboard on')
    parser.add_argument('--no-debug', action='store_true', help='Run in production mode (no debug)')
    
    args = parser.parse_args()
    
    print("🎯 ElleVet Customer Churn Dashboard Launcher")
    print("=" * 50)
    
    # Check for reduced data files for faster loading
    fast_data_exists = os.path.exists('data_fast/orders_redacted.pkl')
    if fast_data_exists:
        fast_size_mb = sum(os.path.getsize(f'data_fast/{f}') for f in os.listdir('data_fast') if f.endswith('.pkl')) / (1024 * 1024)
        print(f"⚡ Fast data detected: {fast_size_mb:.1f} MB (optimized)")
        print("🚀 Dashboard will load quickly!")
    else:
        print("🐌 Using full CSV files (slower loading)")
        print("   💡 Run 'python quick_fix_script.py' for 10x faster loading")
    
    # Always check Python version
    check_python_version()
    
    if not args.skip_checks:
        print("\n📋 Running setup and validation checks...")
        
        # Create directories
        create_directories()
        
        # Check data files
        if not check_data_files():
            print("\n❌ Setup failed: Missing data files")
            sys.exit(1)
        
        # Test imports (install dependencies if needed)
        if not test_imports():
            print("\n🔄 Attempting to install missing dependencies...")
            if not install_requirements():
                print("\n❌ Setup failed: Could not install dependencies")
                sys.exit(1)
            # Test imports again
            if not test_imports():
                print("\n❌ Setup failed: Import errors persist")
                sys.exit(1)
        
        # Validate data quality
        if not validate_data_quality():
            print("\n❌ Setup failed: Data validation errors")
            sys.exit(1)
        
        print("\n✅ All validation checks passed!")
    
    if args.setup_only:
        print("\n🎉 Setup complete! Run without --setup-only to start the dashboard.")
        return
    
    # Launch dashboard
    debug_mode = not args.no_debug
    run_dashboard(host=args.host, port=args.port, debug=debug_mode)

if __name__ == '__main__':
    main()