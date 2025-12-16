# 🛒 Amazon India – A Decade of Sales Analytics 📈🇮🇳

## 📌 Project Overview
This project performs **end-to-end data analytics** on **Amazon India sales data spanning a decade**.  
It covers **data cleaning, database integration, exploratory data analysis (EDA), and business insights generation** using Python and MySQL.

The project demonstrates how raw e-commerce data can be transformed into **actionable business insights**.

---

## 🎯 Objectives
- Clean and preprocess large-scale e-commerce sales data
- Handle missing values, duplicates, and outliers
- Store cleaned data in a **MySQL database**
- Perform **advanced Exploratory Data Analysis (EDA)**
- Generate meaningful **business insights & visualizations**
- Build a reusable analytics pipeline

---

## 🧠 Technologies Used
- **Python**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **MySQL**
- **python-dotenv**
- **Streamlit** (optional)

---


## 📂 Project Structure

- **Amazon_Sales_Analytics/**
  - **data/**
    - raw/ – Raw Amazon sales CSV files  
    - cleaned/ – Cleaned & processed datasets  
  - **scripts/**
    - data_cleaning.py – Data preprocessing & cleaning  
    - database_setup.py – MySQL database & table creation  
    - eda_analysis.py – 20+ EDA & business analysis  
    - config.py – Database configuration  
  - **plots/** – Auto-generated EDA visualizations  
  - **logs/**
    - pipeline_errors.log – Error & execution logs  
  - .env – Database credentials  
  - requirements.txt – Python dependencies  
  - main.py – Main execution file  
  - nan_report.txt – Missing value analysis  
  - README.md – Project documentation


---

## 🗃️ Dataset Overview
- **Rows:** 1,127,000+
- **Columns:** 34
- **Data Type:** Amazon India transaction data
- **Period Covered:** 2015 – 2025

### Key Columns
- order_date  
- customer_id  
- product_id  
- category, subcategory, brand  
- original_price_inr, discounted_price_inr  
- final_amount_inr  
- payment_method  
- delivery_days  
- festival_name  
- is_prime_member  
- return_status  
- customer_rating  

---

## 🧹 Data Cleaning Features
- Date standardization
- Currency & numeric value cleaning
- Missing value handling (mean, median, defaults)
- City & category standardization
- Boolean normalization
- Duplicate detection
- Outlier handling
- Rating normalization (1–5 scale)

---

## 🧮 Database Integration
- Uses **MySQL**
- Automatically:
  - Creates database
  - Creates tables
  - Inserts cleaned data

### `.env` Configuration Example
- DB_HOST=localhost
- DB_USER=root
- DB_PASSWORD=your_password
- DB_NAME=amazon_sales
- DB_PORT=3306


---

## 📊 Exploratory Data Analysis (EDA)
The project answers **20+ business questions**, including:

1. Revenue trends over years  
2. Seasonal sales patterns  
3. Customer segmentation (RFM)  
4. Payment method evolution  
5. Category & subcategory performance  
6. Prime vs Non-Prime behavior  
7. City & state-wise sales  
8. Festival sales impact  
9. Customer age group analysis  
10. Price vs demand analysis  
11. Delivery performance  
12. Return patterns  
13. Brand performance  
14. Customer lifetime value (CLV)  
15. Discount effectiveness  
16. Rating impact on sales  
17. Prime membership growth  
18. Customer segmentation  
19. Geographic expansion opportunities  
20. Business health dashboard  

📈 All results are saved automatically as **high-quality plots**

---

## ⚙️ How to Run the Project

### Step 1: Clone the Repository
- git clone <repository-url>
- cd Amazon_Sales_Analytics

---

### Step 2: Create Virtual Environment (Optional)
- python -m venv venv
- venv\Scripts\activate # Windows
- source venv/bin/activate # Mac/Linux

---

### Step 3: Install Dependencies

- pip install -r requirements.txt

---

### Step 4: Add Raw Data
- Place all Amazon sales CSV files inside:
data/raw/

Example:
- amazon_india_2015.csv
- amazon_india_2016.csv

---

### Step 5: Configure Database
- Update `.env` with your MySQL credentials  
- Ensure MySQL server is running.

---

### Step 6: Run the Project
- python main.py

---

## 🔄 Execution Flow
## 🔄 Project Workflow

Raw Data  
⬇️  
Data Cleaning  
⬇️  
Cleaned Dataset  
⬇️  
MySQL Database  
⬇️  
EDA & Visualizations  
⬇️  
Business Insights  


---

## 📁 Output
- Cleaned CSV file
- MySQL database tables
- 20+ EDA visualizations
- Logs for error tracking

---

## 💼 Real-World Applications
- E-commerce analytics
- Customer behavior analysis
- Pricing & discount strategy
- Marketing campaign analysis
- Supply chain optimization
- Business decision support

---

## 👨‍🎓 Author
**Name:** Saahil Ahamad  
**Batch:** AIML-C-WD-E-B20  



⭐ *If you find this project useful, feel free to star the repository!* ⭐
