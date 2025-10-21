import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

# Opt into future pandas behavior to suppress downcasting warning
pd.set_option('future.no_silent_downcasting', True)

class AmazonDataCleaner:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.df = self.load_data()

    def load_data(self):
        print("Loading transaction data...")
        year_files = [f for f in os.listdir(self.input_dir) if f.startswith('amazon_india_20') and f.endswith('.csv')]
        if not year_files:
            raise FileNotFoundError("No transaction CSV files found in data/raw/")
        dfs = []
        for file in year_files:
            file_path = os.path.join(self.input_dir, file)
            df = pd.read_csv(file_path)
            dfs.append(df)
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(dfs)} files with total shape: {combined_df.shape}")
        return combined_df

    def question_1_clean_dates(self):
        """Question 1: Clean and standardize dates to YYYY-MM-DD."""
        print("Cleaning dates...")
        def parse_date(value):
            try:
                if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                    value = float(value)
                    if value > 1e15:  # Handle Unix timestamps in milliseconds
                        value /= 1e9
                    return pd.to_datetime(value, unit='s', errors='coerce')
                return pd.to_datetime(value, errors='coerce', format='mixed')
            except:
                return pd.NaT
        self.df['order_date'] = self.df['order_date'].apply(parse_date).dt.strftime('%Y-%m-%d')
        invalid_dates = self.df['order_date'].isna().sum()
        if invalid_dates > 0:
            print(f"Dropping {invalid_dates} rows with invalid dates")
            self.df = self.df[self.df['order_date'].notna()]
        self.df['order_year'] = pd.to_datetime(self.df['order_date']).dt.year
        self.df['order_month'] = pd.to_datetime(self.df['order_date']).dt.month
        self.df['order_quarter'] = pd.to_datetime(self.df['order_date']).dt.quarter

    def question_2_clean_prices(self):
        """Question 2: Clean price columns to numeric."""
        print("Cleaning prices...")
        def clean_price(value):
            if isinstance(value, str):
                value = re.sub(r'[₹,\s]', '', value)
                if value.lower() in ['priceonrequest', 'na', '']:
                    return np.nan
                return pd.to_numeric(value, errors='coerce')
            return value
        price_cols = ['original_price_inr', 'discounted_price_inr', 'subtotal_inr', 'delivery_charges', 'final_amount_inr']
        for col in price_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(clean_price)
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)

    def question_3_clean_ratings_and_age(self):
        """Question 3: Standardize customer and product ratings to 1.0-5.0 and handle customer_age_group NaN."""
        print("Cleaning ratings and customer_age_group...")
        def standardize_rating(value):
            if pd.isna(value):
                return np.nan
            if isinstance(value, str):
                value = value.lower().replace('stars', '').replace('star', '').strip()
                if '/' in value:
                    try:
                        num, den = map(float, value.split('/'))
                        return (num / den) * 5 if den != 0 else np.nan
                    except:
                        return np.nan
            return pd.to_numeric(value, errors='coerce')
        for col in ['customer_rating', 'product_rating']:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(standardize_rating).clip(1, 5)
                mean_val = self.df[col].mean()
                self.df[col] = self.df[col].fillna(mean_val)
        # Handle customer_age_group NaN
        if 'customer_age_group' in self.df.columns:
            nan_count = self.df['customer_age_group'].isna().sum()
            print(f"Found {nan_count} NaN values in customer_age_group ({nan_count/len(self.df)*100:.2f}%)")
            self.df['customer_age_group'] = self.df['customer_age_group'].fillna('Unknown')

    def question_4_clean_cities(self):
        """Question 4: Standardize city names."""
        print("Cleaning city names...")
        city_map = {
# Delhi variations
        'delhi': 'Delhi',
        'new delhi': 'Delhi',
        'delhi ncr': 'Delhi',
        
        # Mumbai variations
        'mumbai': 'Mumbai',
        'mumbai ': 'Mumbai',
        'mumba': 'Mumbai',
        'bombay': 'Mumbai',
        
        # Kolkata variations
        'kolkata': 'Kolkata',
        'kolkata ': 'Kolkata',
        'calcutta': 'Kolkata',
        
        # Chennai variations
        'chennai': 'Chennai',
        'chenai': 'Chennai',
        'madras': 'Chennai',
        
        # Bangalore variations
        'bangalore': 'Bangalore',
        'banglore': 'Bangalore',
        'bengalore': 'Bangalore',
        'bengaluru': 'Bangalore',
        
        # Other cities (kept as-is)
        'ludhiana': 'Ludhiana',
        'kochi': 'Kochi',
        'aligarh': 'Aligarh',
        'surat': 'Surat',
        'kanpur': 'Kanpur',
        'hyderabad': 'Hyderabad',
        'bareilly': 'Bareilly',
        'vadodara': 'Vadodara',
        'indore': 'Indore',
        'visakhapatnam': 'Visakhapatnam',
        'lucknow': 'Lucknow',
        'pune': 'Pune',
        'bhubaneswar': 'Bhubaneswar',
        'nagpur': 'Nagpur',
        'patna': 'Patna',
        'ahmedabad': 'Ahmedabad',
        'jaipur': 'Jaipur',
        'meerut': 'Meerut',
        'allahabad': 'Allahabad',  # or map to 'Prayagraj'
        'varanasi': 'Varanasi',
        'coimbatore': 'Coimbatore',
        'moradabad': 'Moradabad',
        'saharanpur': 'Saharanpur',
        'chandigarh': 'Chandigarh',
        'gorakhpur': 'Gorakhpur'
        }
        self.df['customer_city'] = self.df['customer_city'].str.lower().str.strip().replace(city_map).str.title()
        self.df['customer_state'] = self.df['customer_state'].str.title().str.strip()

    def question_5_clean_booleans(self):
        """Question 5: Standardize boolean columns."""
        print("Cleaning boolean columns...")
        bool_cols = ['is_prime_member', 'is_prime_eligible', 'is_festival_sale']
        bool_map = {
            'true': True, 'false': False,
            'yes': True, 'no': False,
            '1': True, '0': False,
            'y': True, 'n': False
        }
        for col in bool_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.lower().str.strip().replace(bool_map).astype(bool)
                self.df[col] = self.df[col].fillna(False)

    def question_6_clean_categories(self):
        """Question 6: Standardize category names with specific mapping for variations."""
        print("Cleaning categories with specific standardization...")
        
        # Simple category mapping for electronics variations
        category_mapping = {
            'electronics': 'electronics',
            'electronic': 'electronics',
            'electronicss': 'electronics',
            'electronics accessories': 'electronics'
        }
        
        # Clean and standardize categories
        self.df['category'] = (self.df['category']
                            .astype(str)
                            .str.lower()
                            .str.strip()
                            .str.replace(r'[/&+]+', ' ', regex=True)
                            .str.replace(r'\s+', ' ', regex=True)
                            .str.strip())
        
        # Apply category mapping
        self.df['category'] = self.df['category'].map(category_mapping).fillna(self.df['category'])
        
        # Convert to title case for consistent formatting
        self.df['category'] = self.df['category'].str.title()
        
        # Show results
        category_counts = self.df['category'].value_counts()
        print(f"Found {len(category_counts)} unique categories after cleaning:")
        for category, count in category_counts.items():
            print(f"  - {category}: {count:,} records")
        
        final_category_count = self.df['category'].nunique()
        print(f"✅ Category standardization complete. Final unique categories: {final_category_count}")
        
        return self.df

    def question_7_clean_delivery_days(self):
        """Question 7: Clean delivery_days to numeric."""
        print("Cleaning delivery days...")
        def clean_days(value):
            if pd.isna(value):
                return np.nan
            if isinstance(value, str):
                value = value.lower().strip()
                if 'same day' in value:
                    return 0.0
                if '1-2 days' in value:
                    return 1.5
                return pd.to_numeric(re.sub(r'[^0-9.]', '', value), errors='coerce')
            return value
        self.df['delivery_days'] = self.df['delivery_days'].apply(clean_days).clip(0, 30)
        mean_days = self.df['delivery_days'].mean()
        self.df['delivery_days'] = self.df['delivery_days'].fillna(mean_days)

    def question_8_handle_duplicates(self):
        """Question 8: Handle duplicate transactions with bulk order detection"""
        print("Handling duplicates...")
        
        # Identify potential duplicates
        dup_cols = ['customer_id', 'product_id', 'order_date', 'final_amount_inr']
        duplicate_mask = self.df.duplicated(subset=dup_cols, keep=False)
        
        if duplicate_mask.sum() > 0:
            duplicates = self.df[duplicate_mask].copy()
            
            # Analyze time patterns - genuine bulk orders might have slight time differences
            duplicates['order_hour'] = pd.to_datetime(duplicates['order_date']).dt.hour
            time_grouped = duplicates.groupby(dup_cols)['order_hour'].std()
            
            # Bulk orders typically have same timestamp or very close
            bulk_orders = time_grouped[time_grouped <= 1].index  # Within 1 hour
            data_errors = time_grouped[time_grouped > 1].index   # Different times
            
            print(f"Found {len(bulk_orders)} potential bulk orders and {len(data_errors)} data errors")
            
            # Keep first instance of bulk orders, remove data errors
            self.df = self.df.drop_duplicates(subset=dup_cols + ['transaction_id'], keep='first')

    def question_9_handle_outliers(self):
        """Question 9: Handle outliers in price columns with decimal point correction"""
        print("Handling outliers...")
        numerical_cols = ['original_price_inr', 'discounted_price_inr', 'subtotal_inr', 'final_amount_inr', 'delivery_charges']
        
        for col in numerical_cols:
            if col in self.df.columns:
                # First, detect potential decimal point errors (prices 100x higher)
                median_price = self.df[col].median()
                potential_decimal_errors = self.df[col] > (median_price * 50)  # Prices 50x above median
                
                # Correct decimal point errors by dividing by 100
                decimal_error_count = potential_decimal_errors.sum()
                if decimal_error_count > 0:
                    print(f"Correcting {decimal_error_count} potential decimal point errors in {col}")
                    self.df.loc[potential_decimal_errors, col] = self.df.loc[potential_decimal_errors, col] / 100
                
                # Then apply IQR method for remaining outliers
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"Clipped {outliers} outliers in {col}")

    def question_10_clean_payment_methods(self):
        """Question 10: Standardize payment methods and handle festival_name NaN."""
        print("Cleaning payment methods and festival_name...")
        payment_map = {
            'upi': 'UPI', 'phonepe': 'UPI', 'googlepay': 'UPI',
            'credit card': 'Credit Card', 'credit_card': 'Credit Card', 'cc': 'Credit Card',
            'cash on delivery': 'COD', 'cod': 'COD', 'c.o.d': 'COD'
        }
        self.df['payment_method'] = self.df['payment_method'].str.lower().str.strip().replace(payment_map).str.upper()
        # Handle festival_name NaN
        nan_count = self.df['festival_name'].isna().sum()
        print(f"Found {nan_count} NaN values in festival_name ({nan_count/len(self.df)*100:.2f}%)")
        self.df['festival_name'] = self.df['festival_name'].astype(str).replace('nan', 'Regular Day')

    def clean_all(self):
        """Apply all cleaning steps."""
        print("\nStarting comprehensive data cleaning...")
        self.question_1_clean_dates()
        self.question_2_clean_prices()
        self.question_3_clean_ratings_and_age()
        self.question_4_clean_cities()
        self.question_5_clean_booleans()
        self.question_6_clean_categories()
        self.question_7_clean_delivery_days()
        self.question_8_handle_duplicates()
        self.question_9_handle_outliers()
        self.question_10_clean_payment_methods()
        # Recalculate derived fields
        self.df['discounted_price_inr'] = self.df['original_price_inr'] * (1 - self.df['discount_percent'] / 100)
        self.df['subtotal_inr'] = self.df['discounted_price_inr'] * self.df['quantity']
        self.df['final_amount_inr'] = self.df['subtotal_inr'] + self.df['delivery_charges']
        print("Data cleaning completed")
        return self.df