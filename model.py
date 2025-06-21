import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import pickle
import warnings
from datetime import datetime, timedelta

try:
    from xgboost import XGBClassifier
    import shap
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')

class ChurnPredictor:
    """
    Churn prediction model for ElleVet cart customers
    """
    
    def __init__(self, model_path='models/'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.model_performance = {}
        self.last_training_date = None
        self.best_model_name = None
        
        # Try to load existing model
        self.load_model()
    
    def prepare_features(self, data):
        """Prepare features for model training/prediction"""
        
        # Select relevant features for churn prediction (including Jason's advanced features)
        all_possible_features = [
            # Original features
            'total_orders', 'total_spent', 'days_since_first_order', 'days_since_last_order',
            'customer_lifetime_days', 'avg_order_value', 'order_value_std', 'min_order_value',
            'max_order_value', 'avg_shipping', 'total_discounts', 'total_tax',
            'shipping_state', 'primary_channel', 'primary_payment', 'primary_utm_source',
            'primary_utm_medium', 'refund_count', 'total_refunded', 'ticket_count',
            
            # Jason's advanced ticket features
            'has_cancel_tag', 'has_urgent_tag', 'mentions_dosing', 'product_issue', 'num_tickets',
            
            # Jason's UTM risk features
            'utm_risk_flag',
            
            # Jason's product features
            'product_form',
            
            # Jason's quiz features
            'cat_flag', 'stress_flag',
            
            # Jason's refund features
            'has_refund',
            
            # Jason's composite risk score
            'jason_risk_score',
            
            # John's discount and behavioral features
            'ever_used_discount', 'total_discount_amount', 'discount_frequency',
            'escalated_flag', 'site_flag', 'spend_tier_low', 'spend_tier_mid', 'spend_tier_high',
            'discount_refund_interaction', 'site_refund_interaction', 'escalated_discount_interaction',
            
            # Terri's clustering features (optional - may not be available)
            'umap_cluster', 'pca_cluster', 'umap_1', 'umap_2', 'pca_1', 'pca_2'
        ]
        
        # Filter to only include features that actually exist in the data
        feature_columns = [col for col in all_possible_features if col in data.columns]
        
        print(f"🔍 Using {len(feature_columns)} available features out of {len(all_possible_features)} possible")
        
        # Create feature dataframe
        features_df = data[feature_columns].copy()
        
        # Handle missing values (including Jason's features)
        features_df = features_df.fillna({
            # Original features
            'order_value_std': 0,
            'total_discounts': 0,
            'total_tax': 0,
            'total_refunded': 0,
            'refund_count': 0,
            'ticket_count': 0,
            'shipping_state': 'Unknown',
            'primary_channel': 'unknown',
            'primary_payment': 'unknown',
            'primary_utm_source': 'unknown',
            'primary_utm_medium': 'unknown',
            
            # Jason's advanced features
            'has_cancel_tag': 0,
            'has_urgent_tag': 0,
            'mentions_dosing': 0,
            'product_issue': 0,
            'num_tickets': 0,
            'utm_risk_flag': 0,
            'product_form': 'Other',
            'cat_flag': 0,
            'stress_flag': 0,
            'has_refund': 0,
            'jason_risk_score': 0,
            
            # John's behavioral features
            'ever_used_discount': 0,
            'total_discount_amount': 0.0,
            'discount_frequency': 0,
            'escalated_flag': 0,
            'site_flag': 0,
            'spend_tier_low': 0,
            'spend_tier_mid': 1,  # Default to middle tier
            'spend_tier_high': 0,
            'discount_refund_interaction': 0,
            'site_refund_interaction': 0,
            'escalated_discount_interaction': 0,
            'spend_tier': 'Mid',  # Default spend tier
            
            # Terri's clustering features
            'umap_cluster': 0,
            'pca_cluster': 0,
            'umap_1': 0,
            'umap_2': 0,
            'pca_1': 0,
            'pca_2': 0
        })
        
        # Encode categorical variables (including Jason's product_form and John's spend_tier)
        categorical_columns = ['shipping_state', 'primary_channel', 'primary_payment', 
                             'primary_utm_source', 'primary_utm_medium', 'product_form', 'spend_tier']
        
        # Only encode categorical columns that actually exist in the features
        existing_categorical_columns = [col for col in categorical_columns if col in features_df.columns]
        
        for col in existing_categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features_df[col] = self.label_encoders[col].fit_transform(features_df[col].astype(str))
            else:
                # Handle new categories in prediction
                unique_values = features_df[col].unique()
                for val in unique_values:
                    if val not in self.label_encoders[col].classes_:
                        # Add new class
                        self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, val)
                features_df[col] = self.label_encoders[col].transform(features_df[col].astype(str))
        
        # Create additional engineered features
        features_df['avg_days_between_orders'] = np.where(
            features_df['total_orders'] > 1,
            features_df['customer_lifetime_days'] / (features_df['total_orders'] - 1),
            0
        )
        
        features_df['order_frequency'] = features_df['total_orders'] / (features_df['days_since_first_order'] + 1)
        features_df['spend_per_day'] = features_df['total_spent'] / (features_df['days_since_first_order'] + 1)
        features_df['has_refunds'] = (features_df['refund_count'] > 0).astype(int)
        features_df['refund_rate'] = features_df['total_refunded'] / (features_df['total_spent'] + 1)
        
        # Remove infinity and NaN values
        features_df = features_df.replace([np.inf, -np.inf], 0)
        features_df = features_df.fillna(0)
        
        return features_df
    
    def train_model(self, data, retrain=False):
        """Train the churn prediction model"""
        
        if self.model is not None and not retrain:
            print("✅ Model already trained. Use retrain=True to retrain.")
            return
        
        print("🤖 Training churn prediction model...")
        
        # Prepare features and target
        features_df = self.prepare_features(data)
        target = data['is_churned'].values
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, target, test_size=0.2, random_state=42, stratify=target
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble of models (including XGBoost if available)
        models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'lr': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),  # John's preferred approach
            'lr_simple': LogisticRegression(random_state=42, max_iter=1000)  # John's original (no class weighting)
        }
        
        # Add XGBoost if available (Jason's approach)
        if XGBOOST_AVAILABLE:
            models['xgb'] = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="auc",
                random_state=42,
                n_jobs=-1
            )
            print("✅ XGBoost model added to ensemble")
        
        best_score = 0
        best_model = None
        
        for name, model in models.items():
            if name in ['lr', 'lr_simple']:  # Both logistic regression variants need scaling
                model.fit(X_train_scaled, y_train)
                score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc').mean()
            elif name == 'xgb' and XGBOOST_AVAILABLE:
                # XGBoost doesn't need scaling
                model.fit(X_train, y_train)
                score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
            else:
                model.fit(X_train, y_train)
                score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
            
            print(f"   {name.upper()} CV AUC: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                self.model = model
                self.best_model_name = name
        
        # Evaluate on test set
        if isinstance(self.model, LogisticRegression) or self.best_model_name in ['lr', 'lr_simple']:
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            y_pred = self.model.predict(X_test_scaled)
            test_features = X_test_scaled
        else:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred = self.model.predict(X_test)
            test_features = X_test
        
        # Store performance metrics
        self.model_performance = {
            'accuracy': (y_pred == y_test).mean(),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'precision': np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0,
            'recall': np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0,
            'test_size': len(y_test),
            'feature_importance': None,
            'churn_definition_version': '2024_fixed_180_day',  # Track churn definition changes
            'churn_threshold': 180  # Store the actual threshold used
        }
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.model_performance['feature_importance'] = importance_df
        
        # Compute SHAP values for interpretability (Jason's approach)
        if XGBOOST_AVAILABLE and hasattr(self.model, 'predict_proba'):
            try:
                print("🔍 Computing SHAP values for model interpretability...")
                if hasattr(self.model, 'feature_importances_'):  # Tree-based models
                    explainer = shap.TreeExplainer(self.model)
                    shap_values = explainer.shap_values(test_features[:100])  # Sample for performance
                    
                    # Store SHAP values and explainer
                    self.model_performance['shap_explainer'] = explainer
                    self.model_performance['sample_shap_values'] = shap_values
                    self.model_performance['sample_features'] = test_features[:100]
                    
                    print("✅ SHAP values computed successfully")
                else:
                    print("⚠️  SHAP not computed - model type not supported")
            except Exception as e:
                print(f"⚠️  SHAP computation failed: {e}")
                self.model_performance['shap_explainer'] = None
        
        self.last_training_date = datetime.now()
        
        print(f"✅ Model training complete!")
        print(f"   Best model: {type(self.model).__name__}")
        print(f"   Test Accuracy: {self.model_performance['accuracy']:.3f}")
        print(f"   Test AUC: {self.model_performance['auc']:.3f}")
        print(f"   Test Precision: {self.model_performance['precision']:.3f}")
        print(f"   Test Recall: {self.model_performance['recall']:.3f}")
        
        # Save model
        self.save_model()
        
        return self.model_performance
    
    def predict_churn(self, data):
        """Predict churn probability for customers"""
        
        # Check if model needs retraining due to churn definition change
        needs_retrain = (
            self.model is None or 
            self.model_performance.get('churn_definition_version') != '2024_fixed_180_day'
        )
        
        if needs_retrain:
            if self.model is None:
                print("⚠️ No trained model available. Training new model...")
            else:
                print("🔄 Churn definition updated. Retraining model for consistency...")
            self.train_model(data, retrain=True)
        
        # Prepare features
        features_df = self.prepare_features(data)
        
        # Scale features if using logistic regression (John's approach needs scaling)
        if isinstance(self.model, LogisticRegression) or self.best_model_name in ['lr', 'lr_simple']:
            features_scaled = self.scaler.transform(features_df)
            churn_probabilities = self.model.predict_proba(features_scaled)[:, 1]
        else:
            # Tree-based models (RF, GB, XGBoost) don't need scaling
            churn_probabilities = self.model.predict_proba(features_df)[:, 1]
        
        # Create predictions dataframe
        predictions = pd.DataFrame({
            'customer_id': data['customer_id'],
            'churn_probability': churn_probabilities,
            'prediction_date': datetime.now()
        })
        
        return predictions
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model_performance.get('feature_importance') is not None:
            return self.model_performance['feature_importance']
        else:
            return pd.DataFrame({'feature': [], 'importance': []})
    
    def get_model_accuracy(self):
        """Get current model accuracy"""
        if self.model_performance:
            return self.model_performance.get('accuracy', 0.0)
        return 0.0
    
    def get_performance_metrics(self):
        """Get all performance metrics"""
        return self.model_performance
    
    def explain_prediction(self, customer_data, top_n=5):
        """Get SHAP explanation for specific customer predictions (Jason's approach)"""
        if not XGBOOST_AVAILABLE or not hasattr(self, 'model_performance') or self.model_performance.get('shap_explainer') is None:
            return None
        
        try:
            # Prepare features for this customer
            features_df = self.prepare_features(customer_data)
            
            # Get SHAP values
            explainer = self.model_performance['shap_explainer']
            
            if isinstance(self.model, LogisticRegression):
                features_scaled = self.scaler.transform(features_df)
                shap_values = explainer.shap_values(features_scaled)
            else:
                shap_values = explainer.shap_values(features_df)
            
            # Create explanation dataframe
            if len(shap_values.shape) > 1 and shap_values.shape[1] > 1:
                # Binary classification - use positive class
                shap_vals = shap_values[:, 1] if shap_values.shape[1] == 2 else shap_values[:, 0]
            else:
                shap_vals = shap_values.flatten()
            
            explanation_df = pd.DataFrame({
                'feature': self.feature_names,
                'shap_value': shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals,
                'feature_value': features_df.iloc[0].values
            })
            
            # Sort by absolute SHAP value impact
            explanation_df['abs_shap'] = abs(explanation_df['shap_value'])
            explanation_df = explanation_df.sort_values('abs_shap', ascending=False)
            
            return explanation_df.head(top_n)
            
        except Exception as e:
            print(f"⚠️  SHAP explanation failed: {e}")
            return None
    
    def get_feature_impact_summary(self):
        """Get summary of feature importance across the dataset"""
        if not hasattr(self, 'model_performance') or self.model_performance.get('sample_shap_values') is None:
            return self.get_feature_importance()
        
        try:
            shap_values = self.model_performance['sample_shap_values']
            
            # Handle different SHAP value formats
            if len(shap_values.shape) > 2:
                # Multi-class case - take positive class
                shap_vals = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[:, :, 0]
            else:
                shap_vals = shap_values
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            
            feature_impact = pd.DataFrame({
                'feature': self.feature_names,
                'mean_abs_shap': mean_abs_shap
            }).sort_values('mean_abs_shap', ascending=False)
            
            return feature_impact
            
        except Exception as e:
            print(f"⚠️  Feature impact summary failed: {e}")
            return self.get_feature_importance()
    
    def save_model(self):
        """Save trained model to disk"""
        try:
            import os
            os.makedirs(self.model_path, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'performance': self.model_performance,
                'training_date': self.last_training_date,
                'best_model_name': self.best_model_name
            }
            
            with open(f'{self.model_path}churn_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ Model saved to {self.model_path}churn_model.pkl")
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            with open(f'{self.model_path}churn_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            self.model_performance = model_data['performance']
            self.last_training_date = model_data['training_date']
            self.best_model_name = model_data.get('best_model_name', 'unknown')
            
            print("✅ Pre-trained model loaded successfully")
            print(f"   Training date: {self.last_training_date}")
            print(f"   Model accuracy: {self.model_performance.get('accuracy', 0):.3f}")
        except FileNotFoundError:
            print("📝 No pre-trained model found. Will train new model on first prediction.")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
    
    def should_retrain(self, days_threshold=7):
        """Check if model should be retrained based on age"""
        if self.last_training_date is None:
            return True
        
        days_since_training = (datetime.now() - self.last_training_date).days
        return days_since_training >= days_threshold
    
    def get_risk_segments(self, predictions):
        """Segment customers by risk level"""
        
        def risk_level(prob):
            if prob >= 0.7:
                return 'High Risk'
            elif prob >= 0.4:
                return 'Medium Risk'
            else:
                return 'Low Risk'
        
        predictions['risk_level'] = predictions['churn_probability'].apply(risk_level)
        
        segment_summary = predictions.groupby('risk_level').agg({
            'customer_id': 'count',
            'churn_probability': 'mean'
        }).rename(columns={'customer_id': 'customer_count', 'churn_probability': 'avg_churn_prob'})
        
        return segment_summary