# database_setup.py
import mysql.connector
import pandas as pd
import os
from mysql.connector import Error
import logging
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
from pathlib import Path

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseSetup:
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'localhost')
        self.user = os.getenv('DB_USER', 'root')
        self.password = os.getenv('DB_PASSWORD', '')
        self.database = os.getenv('DB_NAME', 'amazon_sales')
        self.port = int(os.getenv('DB_PORT', '3306'))
        self.connection = None
        self.cursor = None
        
        # Use flexible paths - will work regardless of where the script is run from
        try:
            base_dir = Path(__file__).parent.parent
        except:
            base_dir = Path.cwd()
        
        self.cleaned_transaction_file = base_dir / 'data' / 'cleaned' / 'amazon_cleaned.csv'
        self.product_catalog_file = base_dir / 'data' / 'raw' / 'amazon_india_products_catalog.csv'
        
        # Alternative paths if the above don't work
        self.alternative_paths = [
            base_dir / 'data' / 'cleaned' / 'amazon_cleaned.csv',
            Path.cwd() / 'data' / 'cleaned' / 'amazon_cleaned.csv',
            Path.cwd() / 'amazon_cleaned.csv'
        ]

    def debug_setup(self):
        """Run debug checks to identify issues"""
        print("\n" + "="*60)
        print("🔧 DATABASE SETUP DEBUG MODE")
        print("="*60)
        
        # Check environment variables
        print("1. 📋 Environment Variables Check:")
        print(f"   ✅ DB_HOST: {self.host}")
        print(f"   ✅ DB_USER: {self.user}")
        print(f"   ✅ DB_NAME: {self.database}")
        print(f"   ✅ DB_PORT: {self.port}")
        print(f"   ✅ DB_PASSWORD: {'*' * len(self.password) if self.password else '❌ Empty'}")
        
        # Check file existence
        print("\n2. 📁 File Existence Check:")
        main_file_exists = os.path.exists(self.cleaned_transaction_file)
        print(f"   {'✅' if main_file_exists else '❌'} Main transaction file: {self.cleaned_transaction_file}")
        
        product_file_exists = os.path.exists(self.product_catalog_file)
        print(f"   {'✅' if product_file_exists else '❌'} Product catalog file: {self.product_catalog_file}")
        
        # Check alternative paths
        if not main_file_exists:
            print("\n   🔍 Searching for transaction file in alternative locations:")
            for path in self.alternative_paths:
                exists = os.path.exists(path)
                print(f"   {'✅' if exists else '❌'} {path}")
                if exists:
                    self.cleaned_transaction_file = path
                    print(f"   ✅ Using alternative path: {path}")
                    break
        
        # Test database connection
        print("\n3. 🗄️ Database Connection Test:")
        try:
            test_conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                port=self.port,
                connection_timeout=5
            )
            print("   ✅ Database connection successful")
            
            # Check if database exists
            cursor = test_conn.cursor()
            cursor.execute("SHOW DATABASES LIKE %s", (self.database,))
            result = cursor.fetchone()
            print(f"   📊 Database '{self.database}': {'✅ Exists' if result else '❌ Does not exist (will be created)'}")
            
            test_conn.close()
        except Exception as e:
            print(f"   ❌ Database connection failed: {e}")
            if "Access denied" in str(e):
                print("   🔐 Check your DB_USER and DB_PASSWORD in .env file")
            elif "Can't connect" in str(e):
                print("   🌐 Check if MySQL server is running and DB_HOST/DB_PORT are correct")
        
        print("="*60)
        return main_file_exists or any(os.path.exists(path) for path in self.alternative_paths)

    def connect(self):
        """Create database connection with enhanced error handling"""
        try:
            # Connect without specifying a database first
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                port=self.port,
                connection_timeout=10
            )
            self.cursor = self.connection.cursor()
            logger.info("✅ Connected to MySQL server successfully")
            return True
        except Error as e:
            logger.error(f"❌ Error connecting to MySQL: {e}")
            
            # Provide specific troubleshooting guidance
            if "Access denied" in str(e):
                logger.error("🔐 Access denied - Check your DB_USER and DB_PASSWORD in .env file")
            elif "Can't connect" in str(e):
                logger.error("🌐 Connection failed - Check if MySQL server is running")
                logger.error("   On Windows: Run 'net start mysql' as administrator")
                logger.error("   On Mac/Linux: Run 'sudo service mysql start'")
            elif "Unknown database" in str(e):
                logger.info("🗄️ Database doesn't exist - it will be created automatically")
            else:
                logger.error(f"🔧 Unknown connection error: {e}")
            
            return False

    def create_database(self):
        """Create database if it doesn't exist"""
        try:
            self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            self.cursor.execute(f"USE {self.database}")
            logger.info(f"✅ Database '{self.database}' created/selected successfully")
            return True
        except Error as e:
            logger.error(f"❌ Error creating database: {e}")
            return False

    def create_tables(self):
        """Create optimized tables for analytics"""
        try:
            # Drop existing tables if they exist
            drop_tables = [
                "DROP TABLE IF EXISTS transactions",
                "DROP TABLE IF EXISTS products",
                "DROP TABLE IF EXISTS customers", 
                "DROP TABLE IF EXISTS time_dimension"
            ]
            
            for query in drop_tables:
                try:
                    self.cursor.execute(query)
                except Error as e:
                    logger.warning(f"Could not drop table (might not exist): {e}")
                    continue
            
            logger.info("✅ Dropped existing tables")

            # Create tables - UPDATED SCHEMA TO MATCH CLEANED DATA
            create_queries = [
                # Main transactions table
                """
                CREATE TABLE transactions (
                    transaction_id VARCHAR(50) PRIMARY KEY,
                    customer_id VARCHAR(50),
                    product_id VARCHAR(50),
                    order_date DATE,
                    quantity INT,
                    original_price_inr DECIMAL(12,2),
                    discount_percent DECIMAL(5,2),
                    discounted_price_inr DECIMAL(12,2),
                    subtotal_inr DECIMAL(12,2),
                    delivery_charges DECIMAL(10,2),
                    final_amount_inr DECIMAL(12,2),
                    customer_rating DECIMAL(3,2),
                    product_rating DECIMAL(3,2),
                    customer_age_group VARCHAR(50),
                    customer_city VARCHAR(100),
                    customer_state VARCHAR(100),
                    is_prime_member BOOLEAN,
                    is_prime_eligible BOOLEAN,
                    is_festival_sale BOOLEAN,
                    festival_name VARCHAR(100),
                    payment_method VARCHAR(50),
                    delivery_days DECIMAL(5,2),
                    category VARCHAR(100),
                    subcategory VARCHAR(100),
                    brand VARCHAR(100),
                    order_year INT,
                    order_month INT,
                    order_quarter INT,
                    customer_spending_tier VARCHAR(50),
                    return_status BOOLEAN
                )
                """,
                
                # Products table
                """
                CREATE TABLE products (
                    product_id VARCHAR(50) PRIMARY KEY,
                    product_name VARCHAR(255),
                    category VARCHAR(100),
                    subcategory VARCHAR(100),
                    brand VARCHAR(100),
                    base_price_2015 DECIMAL(12,2),
                    weight_kg DECIMAL(8,2),
                    rating DECIMAL(3,2),
                    is_prime_eligible BOOLEAN,
                    launch_year INT,
                    model VARCHAR(100)
                )
                """,
                
                # Customers table
                """
                CREATE TABLE customers (
                    customer_id VARCHAR(50) PRIMARY KEY,
                    customer_city VARCHAR(100),
                    customer_state VARCHAR(100),
                    age_group VARCHAR(50),
                    is_prime_member BOOLEAN,
                    customer_spending_tier VARCHAR(50),
                    total_orders INT DEFAULT 0,
                    total_spent DECIMAL(12,2) DEFAULT 0.0,
                    first_order_date DATE,
                    last_order_date DATE
                )
                """,
                
                # Time dimension table
                """
                CREATE TABLE time_dimension (
                    time_id INT AUTO_INCREMENT PRIMARY KEY,
                    order_date DATE,
                    order_day INT,
                    order_month INT,
                    order_quarter INT,
                    order_year INT,
                    day_name VARCHAR(20),
                    month_name VARCHAR(20),
                    is_weekend BOOLEAN,
                    is_festival_season BOOLEAN,
                    festival_name VARCHAR(100)
                )
                """
            ]
            
            for i, query in enumerate(create_queries):
                try:
                    self.cursor.execute(query)
                    logger.info(f"✅ Created table {i+1}/4")
                except Error as e:
                    logger.error(f"❌ Error creating table {i+1}: {e}")
                    return False
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX idx_transactions_date ON transactions(order_date)",
                "CREATE INDEX idx_transactions_customer ON transactions(customer_id)",
                "CREATE INDEX idx_transactions_product ON transactions(product_id)",
                "CREATE INDEX idx_transactions_city ON transactions(customer_city)",
                "CREATE INDEX idx_customers_city ON customers(customer_city)",
                "CREATE INDEX idx_time_date ON time_dimension(order_date)"
            ]
            
            for index in indexes:
                try:
                    self.cursor.execute(index)
                except Error as e:
                    logger.warning(f"Could not create index: {e}")
            
            self.connection.commit()
            logger.info("✅ All tables and indexes created successfully")
            return True
            
        except Error as e:
            logger.error(f"❌ Error creating tables: {e}")
            return False

    def load_cleaned_transaction_data(self, chunksize=5000):
        """Load cleaned transaction data into database with enhanced validation"""
        try:
            # Find the transaction file
            transaction_file = None
            for path in [self.cleaned_transaction_file] + self.alternative_paths:
                if os.path.exists(path):
                    transaction_file = path
                    break
            
            if not transaction_file:
                logger.error("❌ Cleaned transaction file not found in any location")
                logger.error("💡 Make sure data cleaning step ran successfully first")
                logger.error("   Expected locations:")
                for path in [self.cleaned_transaction_file] + self.alternative_paths:
                    logger.error(f"   - {path}")
                return False
            
            # Check if file is not empty
            file_size = os.path.getsize(transaction_file)
            if file_size == 0:
                logger.error("❌ Cleaned transaction file is empty")
                return False
                
            logger.info(f"📥 Loading cleaned transaction data from {transaction_file} (Size: {file_size:,} bytes)")
            
            # ENHANCED DIAGNOSTIC: Check the CSV file before loading
            print("\n🔍 PRE-LOAD DIAGNOSTIC:")
            sample_df = pd.read_csv(transaction_file, nrows=5)
            if 'is_prime_member' in sample_df.columns:
                print(f"Sample is_prime_member values: {sample_df['is_prime_member'].tolist()}")
                print(f"Sample data types: {sample_df['is_prime_member'].dtype}")
            
            # Read the cleaned CSV file
            chunk_count = 0
            total_rows = 0
            
            for chunk in pd.read_csv(transaction_file, chunksize=chunksize, low_memory=False):
                chunk_count += 1
                logger.info(f"🔄 Processing chunk {chunk_count} with {len(chunk):,} rows")
                
                # Clean the data for database insertion
                chunk = self.clean_dataframe(chunk)
                
                # Insert into transactions table
                if not self.insert_transaction_chunk(chunk):
                    logger.error(f"❌ Failed to insert chunk {chunk_count}")
                    return False
                
                total_rows += len(chunk)
                logger.info(f"✅ Processed {total_rows:,} rows so far")
            
            logger.info(f"🎉 Successfully loaded {total_rows:,} transaction records")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading transaction data: {e}")
            import traceback
            logger.error(f"🔍 Detailed error: {traceback.format_exc()}")
            return False

    def load_product_catalog_data(self):
        """Load product catalog data into database"""
        try:
            if not os.path.exists(self.product_catalog_file):
                logger.warning(f"⚠️ Product catalog file not found: {self.product_catalog_file}")
                logger.info("💡 Continuing without product catalog data")
                return True
            
            logger.info(f"📦 Loading product catalog data from {self.product_catalog_file}")
            
            # Read product catalog
            products_df = pd.read_csv(self.product_catalog_file)
            logger.info(f"Loaded {len(products_df):,} products from catalog")
            
            # Clean the data
            products_df = self.clean_dataframe(products_df)
            
            # Insert into products table
            product_sql = """
            INSERT INTO products 
            (product_id, product_name, category, subcategory, brand, base_price_2015, 
             weight_kg, rating, is_prime_eligible, launch_year, model)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            inserted_count = 0
            error_count = 0
            
            for _, row in products_df.iterrows():
                try:
                    self.cursor.execute(product_sql, (
                        str(row['product_id']) if pd.notnull(row['product_id']) else None,
                        str(row['product_name']) if pd.notnull(row['product_name']) else 'Unknown',
                        str(row['category']) if pd.notnull(row['category']) else 'Unknown',
                        str(row['subcategory']) if pd.notnull(row['subcategory']) else 'Unknown',
                        str(row['brand']) if pd.notnull(row['brand']) else 'Unknown',
                        float(row['base_price_2015']) if pd.notnull(row['base_price_2015']) else 0.0,
                        float(row['weight_kg']) if pd.notnull(row['weight_kg']) else 0.0,
                        float(row['rating']) if pd.notnull(row['rating']) else 0.0,
                        bool(row['is_prime_eligible']) if pd.notnull(row['is_prime_eligible']) else False,
                        int(row['launch_year']) if pd.notnull(row['launch_year']) else 2015,
                        str(row['model']) if pd.notnull(row['model']) else 'Standard'
                    ))
                    inserted_count += 1
                except Error as e:
                    error_count += 1
                    if error_count <= 5:  # Log first 5 errors only
                        logger.warning(f"Could not insert product {row['product_id']}: {e}")
                    continue
            
            self.connection.commit()
            logger.info(f"✅ Inserted {inserted_count:,} products into database")
            if error_count > 0:
                logger.warning(f"⚠️ Failed to insert {error_count:,} products")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading product catalog: {e}")
            logger.info("💡 Continuing without product catalog data")
            return True  # Continue even if product catalog fails

    def clean_dataframe(self, df):
        """Clean dataframe for database insertion - IMPROVED BOOLEAN HANDLING"""
        cleaned_df = df.copy()
        
        # Convert date columns
        date_columns = ['order_date']
        for col in date_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce').dt.strftime('%Y-%m-%d')
        
        # Convert numeric columns
        numeric_columns = [
            'original_price_inr', 'discount_percent', 
            'discounted_price_inr', 'subtotal_inr', 'delivery_charges', 'final_amount_inr',
            'customer_rating', 'product_rating', 'delivery_days', 'quantity'
        ]
        
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').fillna(0)
        
        # IMPROVED BOOLEAN HANDLING (CLEAN VERSION)
        boolean_columns = [
            'is_prime_member', 'is_prime_eligible', 'is_festival_sale', 'return_status'
        ]
        
        for col in boolean_columns:
            if col in cleaned_df.columns:
                # Convert to string first to handle mixed types
                cleaned_df[col] = cleaned_df[col].astype(str)
                
                # More comprehensive boolean mapping
                true_values = ['true', 'yes', '1', 'y', 't', 'true.0', 'yes.0', '1.0']
                false_values = ['false', 'no', '0', 'n', 'f', 'false.0', 'no.0', '0.0']
                
                # Map values
                cleaned_df[col] = cleaned_df[col].str.lower().str.strip()
                cleaned_df.loc[cleaned_df[col].isin(true_values), col] = True
                cleaned_df.loc[cleaned_df[col].isin(false_values), col] = False
                
                # Convert to proper boolean
                cleaned_df[col] = cleaned_df[col].astype(bool)
        
        # Fill NaN values with appropriate defaults
        cleaned_df = cleaned_df.where(pd.notnull(cleaned_df), None)
        
        return cleaned_df

    def insert_transaction_chunk(self, chunk):
        """Insert a chunk of transaction data with enhanced boolean handling"""
        try:
            # Define the columns that exist in BOTH the dataframe and table
            transaction_columns = [
                'transaction_id', 'customer_id', 'product_id', 'order_date',
                'quantity', 'original_price_inr', 'discount_percent',
                'discounted_price_inr', 'subtotal_inr', 'delivery_charges', 'final_amount_inr',
                'customer_rating', 'product_rating', 'customer_age_group', 'customer_city',
                'customer_state', 'is_prime_member', 'is_prime_eligible', 'is_festival_sale',
                'festival_name', 'payment_method', 'delivery_days', 'category', 'subcategory',
                'brand', 'order_year', 'order_month', 'order_quarter', 'customer_spending_tier',
                'return_status'
            ]
            
            # Filter columns that exist in the chunk
            available_columns = [col for col in transaction_columns if col in chunk.columns]
            
            # Prepare SQL statement
            placeholders = ', '.join(['%s'] * len(available_columns))
            columns_str = ', '.join(available_columns)
            sql = f"INSERT IGNORE INTO transactions ({columns_str}) VALUES ({placeholders})"
            
            # Insert rows
            batch_size = 500
            total_inserted = 0
            
            for i in range(0, len(chunk), batch_size):
                batch = chunk.iloc[i:i + batch_size]
                batch_data = []
                
                for _, row in batch.iterrows():
                    row_data = []
                    for col in available_columns:
                        value = row[col]
                        # Enhanced data type handling with better boolean support
                        if pd.isna(value) or value is None:
                            row_data.append(None)
                        elif isinstance(value, (np.int64, np.int32, int)):
                            row_data.append(int(value))
                        elif isinstance(value, (np.float64, np.float32, float)):
                            row_data.append(float(value))
                        elif isinstance(value, (bool, np.bool_)):
                            # Proper boolean handling for MySQL (1/0)
                            row_data.append(1 if value else 0)
                        else:
                            row_data.append(str(value) if value is not None else None)
                    batch_data.append(tuple(row_data))
                
                # Use executemany for batch insertion
                if batch_data:
                    try:
                        self.cursor.executemany(sql, batch_data)
                        self.connection.commit()
                        total_inserted += len(batch_data)
                        # Clean progress logging - only show every 10,000 records
                        if total_inserted % 10000 == 0:
                            logger.info(f"Inserted {total_inserted:,} transactions so far")
                    except Error as e:
                        logger.error(f"❌ Error inserting batch: {e}")
                        # Try individual inserts for debugging
                        self.connection.rollback()
                        individual_success = 0
                        for data in batch_data:
                            try:
                                self.cursor.execute(sql, data)
                                individual_success += 1
                            except Error as e:
                                logger.warning(f"Failed to insert individual row: {e}")
                                continue
                        self.connection.commit()
                        total_inserted += individual_success
                        logger.info(f"✅ Inserted {individual_success:,} individual transactions (Total: {total_inserted:,})")
                
            logger.info(f"✅ Successfully inserted {total_inserted:,} transactions from chunk")
            return True
            
        except Error as e:
            logger.error(f"❌ Error inserting transaction chunk: {e}")
            self.connection.rollback()
            return False

    def create_customer_records(self):
        """Create customer records from transaction data"""
        try:
            logger.info("👥 Creating customer records...")
            
            # Create customers from transaction data
            customer_sql = """
            INSERT INTO customers 
            (customer_id, customer_city, customer_state, age_group, is_prime_member, customer_spending_tier,
             total_orders, total_spent, first_order_date, last_order_date)
            SELECT 
                customer_id,
                MAX(customer_city) as customer_city,
                MAX(customer_state) as customer_state,
                MAX(customer_age_group) as age_group,
                MAX(is_prime_member) as is_prime_member,
                MAX(customer_spending_tier) as customer_spending_tier,
                COUNT(*) as total_orders,
                SUM(final_amount_inr) as total_spent,
                MIN(order_date) as first_order_date,
                MAX(order_date) as last_order_date
            FROM transactions 
            GROUP BY customer_id
            ON DUPLICATE KEY UPDATE
                customer_city = VALUES(customer_city),
                customer_state = VALUES(customer_state),
                age_group = VALUES(age_group),
                is_prime_member = VALUES(is_prime_member),
                customer_spending_tier = VALUES(customer_spending_tier),
                total_orders = VALUES(total_orders),
                total_spent = VALUES(total_spent),
                last_order_date = VALUES(last_order_date)
            """
            
            self.cursor.execute(customer_sql)
            self.connection.commit()
            
            # Get count of created customers
            self.cursor.execute("SELECT COUNT(*) FROM customers")
            customer_count = self.cursor.fetchone()[0]
            
            logger.info(f"✅ Customer records created successfully: {customer_count:,} customers")
            return True
            
        except Error as e:
            logger.error(f"❌ Error creating customer records: {e}")
            return False

    def create_time_dimension(self):
        """Create time dimension table"""
        try:
            logger.info("📅 Creating time dimension records...")
            
            # First, drop existing time_dimension table if it exists
            try:
                self.cursor.execute("DROP TABLE IF EXISTS time_dimension")
            except Error as e:
                logger.warning(f"Could not drop time_dimension table: {e}")
            
            # Recreate time_dimension table
            create_time_table = """
            CREATE TABLE time_dimension (
                time_id INT AUTO_INCREMENT PRIMARY KEY,
                order_date DATE,
                order_day INT,
                order_month INT,
                order_quarter INT,
                order_year INT,
                day_name VARCHAR(20),
                month_name VARCHAR(20),
                is_weekend BOOLEAN,
                is_festival_season BOOLEAN,
                festival_name VARCHAR(100)
            )
            """
            self.cursor.execute(create_time_table)
            
            # Insert data with proper table alias to avoid ambiguity
            time_sql = """
            INSERT IGNORE INTO time_dimension 
            (order_date, order_day, order_month, order_quarter, order_year, 
            day_name, month_name, is_weekend, is_festival_season, festival_name)
            SELECT DISTINCT
                t.order_date,
                DAYOFMONTH(t.order_date),
                MONTH(t.order_date),
                QUARTER(t.order_date),
                YEAR(t.order_date),
                DAYNAME(t.order_date),
                MONTHNAME(t.order_date),
                DAYOFWEEK(t.order_date) IN (1, 7),
                MONTH(t.order_date) IN (10, 11),  # Oct-Nov for Diwali
                CASE WHEN MONTH(t.order_date) IN (10, 11) THEN 'Diwali' ELSE 'Regular' END
            FROM transactions t
            WHERE t.order_date IS NOT NULL
            """
            
            self.cursor.execute(time_sql)
            self.connection.commit()
            
            # Get count of created time records
            self.cursor.execute("SELECT COUNT(*) FROM time_dimension")
            time_count = self.cursor.fetchone()[0]
            
            logger.info(f"✅ Time dimension records created successfully: {time_count:,} records")
            return True
            
        except Error as e:
            logger.error(f"❌ Error creating time dimension: {e}")
            self.connection.rollback()
            return False

    def verify_data_loading(self):
            """Verify that data was loaded correctly"""
            try:
                logger.info("🔍 Verifying data loading...")
                
                tables = ['transactions', 'products', 'customers', 'time_dimension']
                results = {}
                
                for table in tables:
                    self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = self.cursor.fetchone()[0]
                    results[table] = count
                    logger.info(f"📊 {table}: {count:,} records")
                
                # Check data integrity
                self.cursor.execute("""
                    SELECT COUNT(DISTINCT customer_id) as unique_customers,
                        COUNT(DISTINCT product_id) as unique_products,
                        MIN(order_date) as earliest_date,
                        MAX(order_date) as latest_date
                    FROM transactions
                """)
                
                integrity = self.cursor.fetchone()
                logger.info(f"👥 Unique customers: {integrity[0]:,}")
                logger.info(f"📦 Unique products: {integrity[1]:,}")
                logger.info(f"📅 Date range: {integrity[2]} to {integrity[3]}")
                
                return results
                
            except Error as e:
                logger.error(f"❌ Error verifying data: {e}")
                return {}

    def verify_boolean_columns(self):
        """Verify that boolean columns were inserted correctly"""
        try:
            logger.info("🔍 Verifying boolean columns in database...")
            
            # Check is_prime_member distribution
            self.cursor.execute("""
                SELECT is_prime_member, COUNT(*) as count 
                FROM transactions 
                GROUP BY is_prime_member
            """)
            prime_results = self.cursor.fetchall()
            
            logger.info("📊 is_prime_member distribution in database:")
            for result in prime_results:
                status = "PRIME" if result[0] == 1 else "NON-PRIME"
                logger.info(f"   {status}: {result[1]:,} records")
            
            # Check other boolean columns
            boolean_columns = ['is_prime_eligible', 'is_festival_sale', 'return_status']
            for col in boolean_columns:
                self.cursor.execute(f"""
                    SELECT {col}, COUNT(*) as count 
                    FROM transactions 
                    GROUP BY {col}
                """)
                results = self.cursor.fetchall()
                logger.info(f"📊 {col} distribution:")
                for result in results:
                    status = "TRUE" if result[0] == 1 else "FALSE"
                    logger.info(f"   {status}: {result[1]:,} records")
            
            return True
            
        except Error as e:
            logger.error(f"❌ Error verifying boolean columns: {e}")
            return False

    def run_complete_setup(self):
        """Run the complete database setup pipeline"""
        try:
            logger.info("🚀 Starting Amazon India Database Setup...")
            
            # Step 1: Connect to MySQL
            if not self.connect():
                return False
            
            # Step 2: Create database
            if not self.create_database():
                return False
            
            # Step 3: Create tables
            if not self.create_tables():
                return False
            
            # Step 4: Load product catalog
            if not self.load_product_catalog_data():
                logger.warning("⚠️ Product catalog loading failed, but continuing...")
            
            # Step 5: Load cleaned transaction data
            if not self.load_cleaned_transaction_data():
                return False
            
            # Step 6: Verify boolean columns were inserted correctly
            if not self.verify_boolean_columns():
                logger.warning("⚠️ Boolean columns verification had issues, but continuing...")
            
            # Step 7: Create customer records
            if not self.create_customer_records():
                logger.warning("⚠️ Customer records creation failed, but continuing...")
            
            # Step 8: Create time dimension
            if not self.create_time_dimension():
                logger.warning("⚠️ Time dimension creation failed, but continuing...")
            
            # Step 9: Verify data loading
            results = self.verify_data_loading()
            
            logger.info("🎉 DATABASE SETUP COMPLETED SUCCESSFULLY!")
            logger.info("📊 Your Amazon India analytics database is ready for PowerBI/Streamlit dashboards!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            import traceback
            logger.error(f"🔍 Detailed error: {traceback.format_exc()}")
            return False
        finally:
            self.close_connection()

    def close_connection(self):
            """Close database connection"""
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
                logger.info("✅ Database connection closed")
   

def main():
    """Main function to run the database setup with debugging"""
    db_setup = DatabaseSetup()
    
    # Run debug first to identify issues
    files_exist = db_setup.debug_setup()
    
    if not files_exist:
        print("\n❌ Required data files not found!")
        print("💡 Please run the data cleaning step first:")
        print("   python main.py")
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Ask user if they want to continue
    response = input("\nDo you want to continue with database setup? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        return
    
    success = db_setup.run_complete_setup()
    
    if success:
        print("\n🎉 DATABASE SETUP COMPLETED SUCCESSFULLY!")
        print("📊 Your data is now ready for analytics and dashboard creation!")
        print("\nNext steps:")
        print("1. 📈 Connect PowerBI/Streamlit to your MySQL database")
        print("2. 🔍 Run exploratory data analysis")
        print("3. 📱 Create interactive dashboards")
    else:
        print("\n❌ Database setup failed. Please check the logs above.")
        print("\n🔧 Troubleshooting tips:")
        print("1. 🗄️ Check if MySQL server is running")
        print("2. 🔐 Verify .env file has correct database credentials")
        print("3. 📁 Ensure data cleaning step completed successfully")
        print("4. ⚙️ Check if you have necessary permissions on the database")
        print("5. 🔄 Try running: python main.py (to ensure data is cleaned first)")

if __name__ == "__main__":
    main()