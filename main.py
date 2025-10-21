import os
from scripts.data_cleaning import AmazonDataCleaner
from scripts.database_setup import DatabaseSetup
from scripts.eda_analysis import EDAAnalysis
from dotenv import load_dotenv

def main():
    print("============================================================\nAmazon India Sales Analytics Pipeline\n============================================================")
    
    # Load environment variables
    load_dotenv()
    
    # Define paths
    raw_data_dir = r'C:\Users\sksaa\OneDrive\Desktop\project 2\data\raw'
    cleaned_data_dir = r'C:\Users\sksaa\OneDrive\Desktop\project 2\data\cleaned'
    product_data_path = r'C:\Users\sksaa\OneDrive\Desktop\project 2\data\raw\amazon_india_products_catalog.csv'
    plots_dir = r'C:\Users\sksaa\OneDrive\Desktop\project 2\plots'
    os.makedirs(cleaned_data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Step 1: Data Cleaning
    print("\nStep 1: Data Cleaning")
    cleaner = AmazonDataCleaner(raw_data_dir)
    cleaned_df = cleaner.clean_all()
    cleaned_data_path = os.path.join(cleaned_data_dir, 'amazon_cleaned.csv')
    cleaned_df.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned data saved to {cleaned_data_path}")
    
    # Step 2: Database Setup
    print("\nStep 2: Database Setup")
    try:
        # Use the default constructor that reads from environment variables
        db = DatabaseSetup()
        success = db.run_complete_setup()
        if success:
            print("✅ Database setup completed successfully!")
        else:
            print("❌ Database setup had some issues, but continuing...")
    except Exception as e:
        print(f"Warning: Database setup failed ({e}). Continuing without database.")
    
    # Step 3: Exploratory Data Analysis
    print("\nStep 3: Exploratory Data Analysis")
    try:
        eda = EDAAnalysis(cleaned_data_path, product_data_path, plots_dir)
        eda.run_all_analysis()
        print("✅ EDA completed successfully!")
    except Exception as e:
        print(f"EDA failed: {e}")
        # Continue with what completed successfully

if __name__ == "__main__":
    main()