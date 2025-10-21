import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import logging
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Use non-interactive backend

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EDAAnalysis:
    def __init__(self, data_path, product_data_path, output_dir):
        self.df = pd.read_csv(data_path, low_memory=False)
        self.product_df = pd.read_csv(product_data_path, low_memory=False)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.insights = []
        # Ensure order_date is datetime
        self.df['order_date'] = pd.to_datetime(self.df['order_date'], errors='coerce')
        logger.info(f"Loaded data with shape: {self.df.shape}")

    def q1_revenue_trend(self):
        """Question 1: Comprehensive revenue trend analysis with growth rates and annotations"""
        logger.info("EDA Question 1: Comprehensive revenue trend analysis")
        
        # Calculate yearly revenue
        yearly_revenue = self.df.groupby('order_year')['final_amount_inr'].sum()
        
        # Calculate growth rates
        yearly_revenue_growth = yearly_revenue.pct_change() * 100
        
        plt.figure(figsize=(12, 8))
        
        # Plot revenue trend
        ax1 = plt.gca()
        line = ax1.plot(yearly_revenue.index, yearly_revenue.values, marker='o', linewidth=2, 
                    markersize=8, color='#1f77b4', label='Revenue')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Total Revenue (INR)', color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax1.grid(True, alpha=0.3)
        
        # Add growth rate as secondary axis
        ax2 = ax1.twinx()
        bars = ax2.bar(yearly_revenue.index, yearly_revenue_growth.values, 
                    alpha=0.3, color='#ff7f0e', label='Growth Rate (%)')
        ax2.set_ylabel('Growth Rate (%)', color='#ff7f0e')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        
        # Add annotations for key points
        max_revenue_year = yearly_revenue.idxmax()
        max_revenue = yearly_revenue.max()
        max_growth_year = yearly_revenue_growth.idxmax()
        max_growth = yearly_revenue_growth.max()
        
        # Annotate peak revenue year
        ax1.annotate(f'Peak: ₹{max_revenue:,.0f}', 
                    xy=(max_revenue_year, max_revenue),
                    xytext=(max_revenue_year, max_revenue * 1.05),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    ha='center', fontweight='bold')
        
        # Annotate highest growth year
        if not pd.isna(max_growth):
            ax1.annotate(f'Max Growth: {max_growth:.1f}%', 
                        xy=(max_growth_year, yearly_revenue.loc[max_growth_year]),
                        xytext=(max_growth_year, yearly_revenue.loc[max_growth_year] * 0.9),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        ha='center', fontweight='bold')
        
        plt.title('Comprehensive Yearly Revenue Trend Analysis (2015-2025)\n.', fontsize=14, fontweight='bold')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'q1_revenue_trend.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def q2_seasonal_patterns(self):
        """Question 2: Analyze seasonal patterns with monthly sales heatmaps"""
        logger.info("EDA Question 2: Seasonal patterns analysis")
        
        # Create month-year combinations for heatmap
        self.df['order_month_num'] = self.df['order_date'].dt.month
        monthly_revenue = self.df.groupby(['order_year', 'order_month_num'])['final_amount_inr'].sum().unstack()
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Plot 1: Monthly revenue patterns across years (line plot)
        years = monthly_revenue.index
        months = monthly_revenue.columns
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for year in years:
            ax1.plot(months, monthly_revenue.loc[year], marker='o', label=year, linewidth=2)
        
        ax1.set_title('Monthly Revenue Patterns Across Years', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Revenue (INR)')
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(month_names, rotation=45)
        ax1.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Heatmap of monthly revenue
        # Normalize data for better heatmap visualization
        heatmap_data = monthly_revenue.fillna(0)
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                cbar_kws={'label': 'Revenue (INR)'}, ax=ax2)
        ax2.set_title('Monthly Revenue Heatmap (Year × Month)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Year')
        ax2.set_xticklabels(month_names, rotation=0)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'q2_seasonal_patterns.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def q3_rfm_segmentation(self):
        """Question 3: Customer segmentation using RFM methodology"""
        logger.info("EDA Question 3: RFM customer segmentation")
        
        try:
            # Define reference date
            reference_date = datetime(2025, 9, 30)
            
            # Calculate RFM metrics
            rfm = self.df.groupby('customer_id').agg({
                'order_date': lambda x: (reference_date - x.max()).days,  # Recency
                'transaction_id': 'count',  # Frequency
                'final_amount_inr': 'sum'   # Monetary
            }).rename(columns={
                'order_date': 'recency',
                'transaction_id': 'frequency',
                'final_amount_inr': 'monetary'
            })
            
            # Remove customers with invalid data
            rfm = rfm[(rfm['recency'] >= 0) & (rfm['frequency'] > 0) & (rfm['monetary'] > 0)]
            
            # Create RFM scores with robust binning
            def create_rfm_scores(data, column, ascending=False):
                """Create RFM scores with fallback for duplicate bins"""
                try:
                    if ascending:
                        return pd.qcut(data[column], 4, labels=[1, 2, 3, 4], duplicates='drop')
                    else:
                        return pd.qcut(data[column], 4, labels=[4, 3, 2, 1], duplicates='drop')
                except ValueError:
                    # Fallback: use custom bins based on data distribution
                    if column == 'recency':
                        bins = [data[column].min()-1, 30, 90, 180, data[column].max()]
                        return pd.cut(data[column], bins=bins, labels=[4, 3, 2, 1])
                    elif column == 'frequency':
                        bins = [data[column].min()-1, 1, 2, 4, data[column].max()]
                        return pd.cut(data[column], bins=bins, labels=[1, 2, 3, 4])
                    else:  # monetary
                        bins = [data[column].min()-1, data[column].quantile(0.25), data[column].quantile(0.5), 
                            data[column].quantile(0.75), data[column].max()]
                        return pd.cut(data[column], bins=bins, labels=[1, 2, 3, 4])
            
            rfm['r_score'] = create_rfm_scores(rfm, 'recency', ascending=False)
            rfm['f_score'] = create_rfm_scores(rfm, 'frequency', ascending=True)
            rfm['m_score'] = create_rfm_scores(rfm, 'monetary', ascending=True)
            
            rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
            
            # Create customer segments based on RFM scores
            def get_rfm_segment(row):
                try:
                    r_score = int(row['r_score'])
                    f_score = int(row['f_score'])
                    m_score = int(row['m_score'])
                    
                    if r_score >= 4 and f_score >= 4 and m_score >= 4:
                        return 'Champions'
                    elif r_score >= 3 and f_score >= 3:
                        return 'Loyal Customers'
                    elif r_score >= 4:
                        return 'New Customers'
                    elif r_score >= 2 and f_score >= 2:
                        return 'Potential Loyalists'
                    elif r_score >= 3:
                        return 'At Risk'
                    else:
                        return 'Lost Customers'
                except (ValueError, TypeError):
                    return 'Unknown'
            
            rfm['segment'] = rfm.apply(get_rfm_segment, axis=1)
            
            # Create visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: RFM Score Distribution
            rfm['total_rfm'] = rfm['r_score'].astype(int) + rfm['f_score'].astype(int) + rfm['m_score'].astype(int)
            sns.histplot(data=rfm, x='total_rfm', bins=20, ax=ax1, color='#2ca02c')
            ax1.set_title('RFM Total Score Distribution')
            ax1.set_xlabel('RFM Total Score')
            ax1.set_ylabel('Number of Customers')
            
            # Plot 2: Customer Segments
            segment_counts = rfm['segment'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
            ax2.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
            ax2.set_title('Customer Segments Distribution')
            
            # Plot 3: Scatter plot - Recency vs Frequency
            scatter = ax3.scatter(rfm['recency'], rfm['frequency'], c=rfm['monetary'], 
                                cmap='viridis', alpha=0.6, s=50)
            ax3.set_xlabel('Recency (Days)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Recency vs Frequency (Color: Monetary)')
            plt.colorbar(scatter, ax=ax3, label='Monetary Value')
            
            # Plot 4: Segment-wise monetary value
            segment_monetary = rfm.groupby('segment')['monetary'].mean().sort_values(ascending=False)
            segment_monetary.plot(kind='bar', ax=ax4, color='#ff7f0e')
            ax4.set_title('Average Monetary Value by Segment')
            ax4.set_xlabel('Customer Segment')
            ax4.set_ylabel('Average Monetary Value (INR)')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'q3_rfm_segmentation.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate insights
            top_segment = segment_counts.index[0]
            champions_count = segment_counts.get('Champions', 0)
            champions_revenue = rfm[rfm['segment'] == 'Champions']['monetary'].sum() if 'Champions' in rfm['segment'].values else 0
            total_revenue = rfm['monetary'].sum()
            champions_share = (champions_revenue / total_revenue) * 100 if total_revenue > 0 else 0
            
            insight = (f"Q3 Insight: {top_segment} is the largest segment ({segment_counts[top_segment]} customers). "
                    f"Champions segment ({champions_count} customers) contributes {champions_share:.1f}% of total revenue.")
            
            self.insights.append(insight)
            
        except Exception as e:
            logger.error(f"RFM segmentation failed: {e}")
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
            self.insights.append("Q3 Insight: RFM segmentation failed due to data issues.")

    def q4_payment_method_evolution(self):
        """Question 4: Evolution of payment methods with stacked area charts"""
        logger.info("EDA Question 4: Payment method evolution analysis")
        
        # Group by year and payment method
        payment_trends = self.df.groupby(['order_year', 'payment_method'])['transaction_id'].count().unstack(fill_value=0)
        
        # Calculate percentages for stacked area chart
        payment_percentages = payment_trends.div(payment_trends.sum(axis=1), axis=0) * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Plot 1: Stacked area chart (market share)
        payment_percentages.plot(kind='area', ax=ax1, stacked=True, alpha=0.7, 
                               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax1.set_title('Payment Method Market Share Evolution (2015-2025)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Market Share (%)')
        ax1.legend(title='Payment Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Line chart (absolute numbers)
        payment_trends.plot(kind='line', ax=ax2, marker='o', linewidth=2)
        ax2.set_title('Payment Method Transaction Volume Trends', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Number of Transactions')
        ax2.legend(title='Payment Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'q4_payment_method_evolution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate insights
        latest_year = payment_trends.index.max()
        latest_data = payment_trends.loc[latest_year]
        dominant_method = latest_data.idxmax()
        dominant_share = (latest_data.max() / latest_data.sum()) * 100
        
        # Calculate UPI growth if available
        upi_methods = [col for col in payment_trends.columns if 'upi' in str(col).lower() or 'phonepe' in str(col).lower() or 'google' in str(col).lower()]
        if upi_methods:
            upi_growth = (payment_trends[upi_methods].sum(axis=1).iloc[-1] / payment_trends[upi_methods].sum(axis=1).iloc[0]) * 100
        
        # Find COD decline
        cod_methods = [col for col in payment_trends.columns if 'cod' in str(col).lower() or 'cash' in str(col).lower()]
        if cod_methods:
            cod_decline = ((payment_trends[cod_methods].sum(axis=1).iloc[0] - payment_trends[cod_methods].sum(axis=1).iloc[-1]) / 
                          payment_trends[cod_methods].sum(axis=1).iloc[0]) * 100
        
        insight = (f"Q4 Insight: {dominant_method} dominates with {dominant_share:.1f}% market share in {latest_year}. "
                  f"Digital payment methods show significant growth over the decade.")
        
        self.insights.append(insight)

    def q5_subcategory_performance(self):
        """Question 5: Subcategory-wise performance analysis with multiple visualization types"""
        logger.info("EDA Question 5: Subcategory performance analysis")
        
        # Calculate subcategory metrics
        subcategory_revenue = self.df.groupby('subcategory')['final_amount_inr'].sum().sort_values(ascending=False)
        subcategory_quantity = self.df.groupby('subcategory')['quantity'].sum().sort_values(ascending=False)
        subcategory_growth = self.df.groupby(['order_year', 'subcategory'])['final_amount_inr'].sum().unstack().pct_change().mean() * 100
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Plot 1: Treemap (simulated with bar chart)
        colors = plt.cm.Set3(np.linspace(0, 1, len(subcategory_revenue)))
        bars = ax1.bar(range(len(subcategory_revenue)), subcategory_revenue.values, color=colors)
        ax1.set_title('Revenue by Subcategory (Treemap Style)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Subcategory')
        ax1.set_ylabel('Total Revenue (INR)')
        ax1.set_xticks(range(len(subcategory_revenue)))
        ax1.set_xticklabels(subcategory_revenue.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'₹{height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Pie chart for market share
        wedges, texts, autotexts = ax2.pie(subcategory_revenue.values, labels=subcategory_revenue.index, 
                                        autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('Subcategory Market Share', fontsize=14, fontweight='bold')
        
        # Plot 3: Growth rates
        subcategory_growth_sorted = subcategory_growth.reindex(subcategory_revenue.index)
        bars_growth = ax3.bar(range(len(subcategory_growth_sorted)), subcategory_growth_sorted.values, 
                            color=colors, alpha=0.7)
        ax3.set_title('Average Annual Growth Rate by Subcategory', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Subcategory')
        ax3.set_ylabel('Growth Rate (%)')
        ax3.set_xticks(range(len(subcategory_growth_sorted)))
        ax3.set_xticklabels(subcategory_growth_sorted.index, rotation=45, ha='right')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 4: Quantity vs Revenue scatter
        scatter = ax4.scatter(subcategory_quantity, subcategory_revenue, s=200, alpha=0.7, c=colors)
        ax4.set_title('Quantity vs Revenue by Subcategory', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Total Quantity Sold')
        ax4.set_ylabel('Total Revenue (INR)')
        
        # Add subcategory labels to scatter points
        for i, subcategory in enumerate(subcategory_revenue.index):
            ax4.annotate(subcategory, (subcategory_quantity.loc[subcategory], subcategory_revenue.loc[subcategory]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'q5_subcategory_performance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def q6_prime_membership_impact(self):
        """Question 6: Prime membership impact on customer behavior with subcategory preferences"""
        logger.info("EDA Question 6: Prime membership impact analysis")
        
        # Calculate metrics for Prime vs Non-Prime
        prime_metrics = self.df.groupby('is_prime_member').agg({
            'final_amount_inr': ['mean', 'sum', 'count'],
            'quantity': 'mean',
            'customer_id': 'nunique',
            'transaction_id': 'count'
        }).round(2)
        
        # Calculate order frequency
        customer_order_freq = self.df.groupby(['customer_id', 'is_prime_member'])['transaction_id'].count().reset_index()
        order_freq_comparison = customer_order_freq.groupby('is_prime_member')['transaction_id'].mean()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Average Order Value comparison
        aov_comparison = self.df.groupby('is_prime_member')['final_amount_inr'].mean()
        bars1 = ax1.bar(['Non-Prime', 'Prime'], aov_comparison.values, 
                    color=['#ff7f0e', '#1f77b4'], alpha=0.7)
        ax1.set_title('Average Order Value: Prime vs Non-Prime', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Order Value (INR)')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'₹{height:,.0f}', ha='center', va='bottom')
        
        # Plot 2: Order Frequency comparison
        bars2 = ax2.bar(['Non-Prime', 'Prime'], order_freq_comparison.values,
                    color=['#ff7f0e', '#1f77b4'], alpha=0.7)
        ax2.set_title('Average Order Frequency', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Orders per Customer')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # Plot 3: Subcategory preferences
        prime_subcategory = self.df[self.df['is_prime_member'] == True].groupby('subcategory')['final_amount_inr'].sum()
        non_prime_subcategory = self.df[self.df['is_prime_member'] == False].groupby('subcategory')['final_amount_inr'].sum()
        
        # Get top 8 subcategories for better visualization
        top_subcategories = self.df.groupby('subcategory')['final_amount_inr'].sum().nlargest(8).index
        
        prime_subcategory_top = prime_subcategory.reindex(top_subcategories).fillna(0)
        non_prime_subcategory_top = non_prime_subcategory.reindex(top_subcategories).fillna(0)
        
        # Normalize for comparison
        prime_subcategory_pct = (prime_subcategory_top / prime_subcategory_top.sum()) * 100
        non_prime_subcategory_pct = (non_prime_subcategory_top / non_prime_subcategory_top.sum()) * 100
        
        x = np.arange(len(top_subcategories))
        width = 0.35
        
        ax3.bar(x - width/2, non_prime_subcategory_pct.values, width, label='Non-Prime', alpha=0.7, color='#ff7f0e')
        ax3.bar(x + width/2, prime_subcategory_pct.values, width, label='Prime', alpha=0.7, color='#1f77b4')
        ax3.set_title('Subcategory Preferences (%) - Top 8', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Subcategory')
        ax3.set_ylabel('Percentage of Revenue (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(top_subcategories, rotation=45, ha='right')
        ax3.legend()
        
        # Plot 4: Revenue contribution over time
        yearly_prime_revenue = self.df.groupby(['order_year', 'is_prime_member'])['final_amount_inr'].sum().unstack()
        yearly_prime_revenue.plot(kind='line', ax=ax4, marker='o', linewidth=2)
        ax4.set_title('Revenue Trend: Prime vs Non-Prime', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Revenue (INR)')
        ax4.legend(['Non-Prime', 'Prime'])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'q6_prime_membership_impact.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate insights
        prime_aov = aov_comparison[True]
        non_prime_aov = aov_comparison[False]
        prime_freq = order_freq_comparison[True]
        non_prime_freq = order_freq_comparison[False]
        aov_difference = ((prime_aov - non_prime_aov) / non_prime_aov) * 100
        freq_difference = ((prime_freq - non_prime_freq) / non_prime_freq) * 100
        
        prime_preferred_subcategory = prime_subcategory_pct.idxmax()
        non_prime_preferred_subcategory = non_prime_subcategory_pct.idxmax()
        
        insight = (f"Q6 Insight: Prime members spend {aov_difference:+.1f}% more per order (₹{prime_aov:,.0f} vs ₹{non_prime_aov:,.0f}) "
                f"and order {freq_difference:+.1f}% more frequently. Prime members prefer {prime_preferred_subcategory}, "
                f"while non-Prime prefer {non_prime_preferred_subcategory}.")
        
        self.insights.append(insight)

    def q7_geographic_analysis(self):
        """Question 7: Geographic analysis across Indian cities and states"""
        logger.info("EDA Question 7: Geographic sales analysis")
        
        # Calculate city-wise and state-wise metrics
        city_revenue = self.df.groupby('customer_city')['final_amount_inr'].sum().sort_values(ascending=False)
        state_revenue = self.df.groupby('customer_state')['final_amount_inr'].sum().sort_values(ascending=False)
        
        # Classify cities into tiers (simplified classification)
        def classify_city_tier(city_name):
            metro_cities = ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad']
            tier1_cities = ['pune', 'ahmedabad', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'surat']
            
            city_lower = str(city_name).lower()
            if any(metro in city_lower for metro in metro_cities):
                return 'Metro'
            elif any(tier1 in city_lower for tier1 in tier1_cities):
                return 'Tier 1'
            else:
                return 'Tier 2/Rural'
        
        self.df['city_tier'] = self.df['customer_city'].apply(classify_city_tier)
        tier_revenue = self.df.groupby('city_tier')['final_amount_inr'].sum()
        tier_growth = self.df.groupby(['order_year', 'city_tier'])['final_amount_inr'].sum().unstack().pct_change().mean() * 100
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Top 10 cities by revenue
        city_revenue.head(10).plot(kind='bar', ax=ax1, color='#1f77b4', alpha=0.7)
        ax1.set_title('Top 10 Cities by Revenue', fontsize=14, fontweight='bold')
        ax1.set_xlabel('City')
        ax1.set_ylabel('Revenue (INR)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Top 10 states by revenue
        state_revenue.head(10).plot(kind='bar', ax=ax2, color='#ff7f0e', alpha=0.7)
        ax2.set_title('Top 10 States by Revenue', fontsize=14, fontweight='bold')
        ax2.set_xlabel('State')
        ax2.set_ylabel('Revenue (INR)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Revenue by city tier
        tier_revenue.plot(kind='pie', ax=ax3, autopct='%1.1f%%', startangle=90, 
                         colors=['#2ca02c', '#d62728', '#9467bd'])
        ax3.set_title('Revenue Distribution by City Tier', fontsize=14, fontweight='bold')
        ax3.set_ylabel('')
        
        # Plot 4: Growth patterns by tier
        tier_growth.plot(kind='bar', ax=ax4, color=['#2ca02c', '#d62728', '#9467bd'], alpha=0.7)
        ax4.set_title('Average Growth Rate by City Tier', fontsize=14, fontweight='bold')
        ax4.set_xlabel('City Tier')
        ax4.set_ylabel('Growth Rate (%)')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'q7_geographic_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate insights
        top_city = city_revenue.index[0]
        top_city_revenue = city_revenue.iloc[0]
        top_state = state_revenue.index[0]
        top_state_revenue = state_revenue.iloc[0]
        dominant_tier = tier_revenue.idxmax()
        dominant_tier_share = (tier_revenue.max() / tier_revenue.sum()) * 100
        fastest_growing_tier = tier_growth.idxmax()
        fastest_growth = tier_growth.max()
        
        insight = (f"Q7 Insight: {top_city} leads cities (₹{top_city_revenue:,.0f}), {top_state} leads states (₹{top_state_revenue:,.0f}). "
                  f"{dominant_tier} cities dominate with {dominant_tier_share:.1f}% share. "
                  f"{fastest_growing_tier} shows highest growth at {fastest_growth:.1f}%.")
        
        self.insights.append(insight)

    def q8_festival_sales_impact(self):
        """Question 8: Festival sales impact with before/during/after analysis using subcategories"""
        logger.info("EDA Question 8: Festival sales impact analysis")
        
        # Clean festival names and handle NaN values
        self.df['festival_name'] = self.df['festival_name'].fillna('Regular Day')
        self.df['festival_name'] = self.df['festival_name'].replace('nan', 'Regular Day')
        
        # Focus on major festivals - use flexible matching
        major_festivals = ['Republic Day Sale', 'Regular Day', 'Valentine Sale','Holi Festival', 'Summer Sale', 'Back to School', 'Prime Day','Amazon Great Indian Festival', 'Diwali Sale']
        
        # Create festival period analysis with flexible matching
        festival_data = {}
        
        for festival in major_festivals:
            # Use flexible matching for festival names
            festival_mask = self.df['festival_name'].str.contains(festival, case=False, na=False)
            festival_df = self.df[festival_mask]
            
            if not festival_df.empty and len(festival_df) > 10:  # Ensure minimum data points
                festival_revenue = festival_df['final_amount_inr'].sum()
                festival_orders = festival_df['transaction_id'].count()
                festival_data[festival] = {
                    'revenue': festival_revenue,
                    'orders': festival_orders,
                    'aov': festival_revenue / festival_orders if festival_orders > 0 else 0
                }
                print(f"✅ Found {festival}: {festival_orders} orders, ₹{festival_revenue:,.0f} revenue")
            else:
                print(f"❌ Insufficient data for {festival}: {len(festival_df)} records")
        
        # Compare with regular days
        regular_days = self.df[self.df['festival_name'] == 'Regular Day']
        regular_revenue = regular_days['final_amount_inr'].sum()
        regular_orders = regular_days['transaction_id'].count()
        regular_aov = regular_revenue / regular_orders if regular_orders > 0 else 0
        
        festival_data['Regular Day'] = {
            'revenue': regular_revenue,
            'orders': regular_orders,
            'aov': regular_aov
        }
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Revenue by festival
        festival_revenues = [data['revenue'] for data in festival_data.values()]
        festival_names = list(festival_data.keys())
        
        # Use colors based on data availability
        colors = ['#e377c2' if name != 'Regular Day' else '#7f7f7f' for name in festival_names]
        
        bars1 = ax1.bar(festival_names, festival_revenues, color=colors, alpha=0.7)
        ax1.set_title('Total Revenue by Festival Period', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Festival')
        ax1.set_ylabel('Revenue (INR)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:  # Only add labels for non-zero values
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'₹{height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Average Order Value comparison
        festival_aov = [data['aov'] for data in festival_data.values()]
        bars2 = ax2.bar(festival_names, festival_aov, color=colors, alpha=0.7)
        ax2.set_title('Average Order Value by Festival Period', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Festival')
        ax2.set_ylabel('Average Order Value (INR)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels for AOV
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'₹{height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Time series analysis for Diwali (with flexible matching)
        diwali_mask = self.df['festival_name'].str.contains('diwali', case=False, na=False)
        diwali_data = self.df[diwali_mask]
        
        if not diwali_data.empty:
            diwali_years = diwali_data.groupby('order_year')['final_amount_inr'].sum()
            if not diwali_years.empty:
                ax3.plot(diwali_years.index, diwali_years.values, marker='o', color='#ff7f0e', linewidth=2, label='Diwali Sales')
                ax3.set_title('Diwali Sales Trend Over Years', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Year')
                ax3.set_ylabel('Revenue (INR)')
                ax3.grid(True, alpha=0.3)
                
                # Add value labels for Diwali years
                for year, revenue in diwali_years.items():
                    ax3.annotate(f'₹{revenue:,.0f}', (year, revenue), 
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
            else:
                ax3.text(0.5, 0.5, 'Diwali yearly data not available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Diwali Sales Trend (No Yearly Data)', fontsize=14, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Diwali sales data found', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Diwali Sales Trend (Data Not Available)', fontsize=14, fontweight='bold')
        
        # Plot 4: Subcategory performance during festivals vs regular days
        festival_subcategories = self.df[self.df['festival_name'] != 'Regular Day'].groupby('subcategory')['final_amount_inr'].sum()
        regular_subcategories = self.df[self.df['festival_name'] == 'Regular Day'].groupby('subcategory')['final_amount_inr'].sum()
        
        # Get top 5 subcategories for both
        top_festival_subs = festival_subcategories.nlargest(5)
        top_regular_subs = regular_subcategories.nlargest(5)
        
        # Ensure we have common subcategories for comparison
        common_subs = top_festival_subs.index.intersection(top_regular_subs.index)
        
        if len(common_subs) > 0:
            x = np.arange(len(common_subs))
            width = 0.35
            
            regular_values = [top_regular_subs.get(sub, 0) for sub in common_subs]
            festival_values = [top_festival_subs.get(sub, 0) for sub in common_subs]
            
            ax4.bar(x - width/2, regular_values, width, label='Regular Days', alpha=0.7, color='#1f77b4')
            ax4.bar(x + width/2, festival_values, width, label='Festival Days', alpha=0.7, color='#d62728')
            ax4.set_title('Top Subcategories: Festival vs Regular Days', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Subcategory')
            ax4.set_ylabel('Revenue (INR)')
            ax4.set_xticks(x)
            ax4.set_xticklabels(common_subs, rotation=45, ha='right')
            ax4.legend()
            
            # Add value labels
            for i, (reg_val, fest_val) in enumerate(zip(regular_values, festival_values)):
                if reg_val > 0:
                    ax4.text(x[i] - width/2, reg_val, f'₹{reg_val:,.0f}', 
                            ha='center', va='bottom', fontsize=8)
                if fest_val > 0:
                    ax4.text(x[i] + width/2, fest_val, f'₹{fest_val:,.0f}', 
                            ha='center', va='bottom', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'Subcategory comparison data not available\n(Try running data cleaning first)', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Subcategory Comparison (Data Not Available)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'q8_festival_sales_impact.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate insights
        if len(festival_data) > 1:  # More than just Regular Day
            # Remove Regular Day for festival analysis
            festival_only = {k: v for k, v in festival_data.items() if k != 'Regular Day'}
            
            if festival_only:
                top_festival = max(festival_only.items(), key=lambda x: x[1]['revenue'])[0]
                top_festival_revenue = festival_data[top_festival]['revenue']
                top_festival_aov = festival_data[top_festival]['aov']
                
                festival_boost = ((top_festival_aov - regular_aov) / regular_aov) * 100 if regular_aov > 0 else 0
                
                # Subcategory insights
                if len(common_subs) > 0:
                    top_festival_sub = top_festival_subs.index[0] if len(top_festival_subs) > 0 else 'Unknown'
                    top_regular_sub = top_regular_subs.index[0] if len(top_regular_subs) > 0 else 'Unknown'
                    insight = (f"Q8 Insight: {top_festival} generates highest festival revenue (₹{top_festival_revenue:,.0f}). "
                            f"Festival AOV is {festival_boost:+.1f}% higher than regular days. "
                            f"Top festival subcategory: {top_festival_sub}, Top regular: {top_regular_sub}.")
                else:
                    insight = (f"Q8 Insight: {top_festival} generates highest festival revenue (₹{top_festival_revenue:,.0f}). "
                            f"Festival AOV is {festival_boost:+.1f}% higher than regular days.")
            else:
                insight = "Q8 Insight: Limited festival data available for analysis."
        else:
            insight = "Q8 Insight: No festival data found beyond regular days."
        
        self.insights.append(insight)
        
        # Print debug information
        print(f"\n🎯 FESTIVAL ANALYSIS DEBUG:")
        print(f"   Festival names in data: {self.df['festival_name'].unique()}")
        print(f"   Festival data found: {list(festival_data.keys())}")
        print(f"   Diwali records: {len(diwali_data)}")
        if not diwali_data.empty:
            print(f"   Diwali years: {diwali_data['order_year'].unique()}")

    def q9_customer_age_group_analysis(self):
        """Question 9: Analyze customer age group behavior and preferences"""
        logger.info("EDA Question 9: Customer age group analysis")
        
        try:
            # Check if age group column exists
            if 'customer_age_group' not in self.df.columns:
                logger.warning("❌ customer_age_group column not found in dataset")
                self.insights.append("Q9 Insight: Age group data not available for analysis.")
                return
            
            # Clean age group data
            self.df['customer_age_group'] = self.df['customer_age_group'].fillna('Unknown')
            
            # Age group distribution
            age_distribution = self.df['customer_age_group'].value_counts()
            
            # Spending patterns by age group
            age_spending = self.df.groupby('customer_age_group').agg({
                'final_amount_inr': ['mean', 'sum', 'count'],
                'quantity': 'mean',
                'customer_rating': 'mean'
            }).round(2)
            
            # Flatten column names
            age_spending.columns = ['avg_spending', 'total_revenue', 'order_count', 'avg_quantity', 'avg_rating']
            
            # Category preferences by age group
            age_category_pref = pd.crosstab(self.df['customer_age_group'], 
                                        self.df['category'], 
                                        values=self.df['final_amount_inr'], 
                                        aggfunc='sum').fillna(0)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Age group distribution
            colors1 = plt.cm.Set3(np.linspace(0, 1, len(age_distribution)))
            bars1 = ax1.bar(age_distribution.index, age_distribution.values, color=colors1)
            ax1.set_title('Customer Age Group Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Age Group')
            ax1.set_ylabel('Number of Customers')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,}', ha='center', va='bottom')
            
            # Plot 2: Average spending by age group
            bars2 = ax2.bar(age_spending.index, age_spending['avg_spending'].values, 
                        color='lightcoral', alpha=0.7)
            ax2.set_title('Average Spending by Age Group', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Age Group')
            ax2.set_ylabel('Average Spending (INR)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'₹{height:,.0f}', ha='center', va='bottom')
            
            # Plot 3: Category preferences heatmap (top categories only)
            # Get top categories for better visualization
            top_categories = age_category_pref.sum().nlargest(8).index
            age_category_top = age_category_pref[top_categories]
            
            if not age_category_top.empty:
                sns.heatmap(age_category_top, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax3,
                        cbar_kws={'label': 'Revenue (INR)'})
                ax3.set_title('Category Preferences by Age Group (Top Categories)', 
                            fontsize=14, fontweight='bold')
                ax3.set_xlabel('Category')
                ax3.set_ylabel('Age Group')
            else:
                ax3.text(0.5, 0.5, 'Category preference data not available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Category Preferences (Data Not Available)', fontsize=14, fontweight='bold')
            
            # Plot 4: Order frequency by age group
            order_freq = self.df.groupby(['customer_id', 'customer_age_group']).size().reset_index(name='order_count')
            avg_order_freq = order_freq.groupby('customer_age_group')['order_count'].mean()
            
            bars4 = ax4.bar(avg_order_freq.index, avg_order_freq.values, color='lightgreen', alpha=0.7)
            ax4.set_title('Average Order Frequency by Age Group', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Age Group')
            ax4.set_ylabel('Average Orders per Customer')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar in bars4:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'q9_customer_age_group_analysis.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate insights
            largest_age_group = age_distribution.index[0]
            largest_age_count = age_distribution.iloc[0]
            highest_spending_age = age_spending['avg_spending'].idxmax()
            highest_spending = age_spending['avg_spending'].max()
            most_frequent_age = avg_order_freq.idxmax()
            most_frequent_value = avg_order_freq.max()
            
            insight = (f"Q9 Insight: {largest_age_group} is largest group ({largest_age_count:,} customers). "
                    f"{highest_spending_age} spends most (₹{highest_spending:,.0f}/order). "
                    f"{most_frequent_age} orders most frequently ({most_frequent_value:.1f} orders/customer).")
            
            self.insights.append(insight)
            logger.info(f"✅ Age group analysis completed: {largest_age_group} dominant group")
            
        except Exception as e:
            logger.error(f"❌ Age group analysis failed: {e}")
            self.insights.append("Q9 Insight: Age group analysis failed due to data issues.")

    def q10_price_vs_demand_analysis(self):
        """Question 10: Build price vs demand analysis using scatter plots and correlation matrices"""
        logger.info("EDA Question 10: Price vs demand analysis")
        
        try:
            # Create price segments
            price_bins = [0, 500, 1000, 5000, 10000, 50000, float('inf')]
            price_labels = ['<500', '500-1K', '1K-5K', '5K-10K', '10K-50K', '>50K']
            self.df['price_segment'] = pd.cut(self.df['original_price_inr'], 
                                            bins=price_bins,
                                            labels=price_labels)
            
            # Calculate overall price vs quantity correlation
            price_demand_corr = self.df[['original_price_inr', 'quantity']].corr().iloc[0,1]
            
            # Category-wise price elasticity
            category_correlations = []
            for category in self.df['category'].unique():
                category_data = self.df[self.df['category'] == category]
                if len(category_data) > 10:  # Minimum data points
                    corr = category_data[['original_price_inr', 'quantity']].corr().iloc[0,1]
                    category_correlations.append((category, corr))
            
            category_elasticity = pd.DataFrame(category_correlations, 
                                            columns=['category', 'correlation']).set_index('category')
            
            # Price segment performance
            price_segment_perf = self.df.groupby('price_segment').agg({
                'quantity': 'sum',
                'final_amount_inr': 'sum',
                'transaction_id': 'count'
            }).rename(columns={'transaction_id': 'order_count'})
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Price vs Quantity scatter plot
            sample_size = min(5000, len(self.df))  # Sample for better visualization
            sample_df = self.df.sample(sample_size, random_state=42)
            
            scatter = ax1.scatter(sample_df['original_price_inr'], sample_df['quantity'], 
                                alpha=0.6, s=30, c=sample_df['final_amount_inr'], 
                                cmap='viridis', edgecolors='black', linewidth=0.5)
            ax1.set_xlabel('Price (INR)')
            ax1.set_ylabel('Quantity Sold')
            ax1.set_title(f'Price vs Demand Analysis\n(Correlation: {price_demand_corr:.3f})', 
                        fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax1, label='Order Value (INR)')
            
            # Set reasonable limits for better visualization
            ax1.set_xlim(0, self.df['original_price_inr'].quantile(0.95))
            ax1.set_ylim(0, self.df['quantity'].quantile(0.95))
            
            # Plot 2: Price elasticity by category
            if not category_elasticity.empty:
                category_elasticity_sorted = category_elasticity.sort_values('correlation')
                bars2 = ax2.barh(range(len(category_elasticity_sorted)), 
                            category_elasticity_sorted['correlation'].values, 
                            color=plt.cm.RdYlGn_r(np.linspace(0, 1, len(category_elasticity_sorted))))
                ax2.set_yticks(range(len(category_elasticity_sorted)))
                ax2.set_yticklabels(category_elasticity_sorted.index)
                ax2.set_title('Price Elasticity by Category', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Correlation Coefficient\n(Negative = Elastic, Positive = Inelastic)')
                ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                
                # Add correlation values on bars
                for i, (idx, row) in enumerate(category_elasticity_sorted.iterrows()):
                    ax2.text(row['correlation'], i, f'{row["correlation"]:.3f}', 
                            ha='left' if row['correlation'] >= 0 else 'right', 
                            va='center', fontsize=9)
            
            # Plot 3: Price segment distribution
            price_segment_perf['order_count'].plot(kind='bar', ax=ax3, color='skyblue', alpha=0.7)
            ax3.set_title('Order Distribution by Price Segment', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Price Segment (INR)')
            ax3.set_ylabel('Number of Orders')
            ax3.tick_params(axis='x', rotation=45)
            
            # Plot 4: Revenue contribution by price segment
            wedges, texts, autotexts = ax4.pie(price_segment_perf['final_amount_inr'], 
                                            labels=price_segment_perf.index,
                                            autopct='%1.1f%%', startangle=90,
                                            colors=plt.cm.Set3(np.linspace(0, 1, len(price_segment_perf))))
            ax4.set_title('Revenue Contribution by Price Segment', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'q10_price_vs_demand_analysis.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate insights
            most_elastic_category = category_elasticity['correlation'].idxmin() if not category_elasticity.empty else 'Unknown'
            most_elastic_value = category_elasticity['correlation'].min() if not category_elasticity.empty else 0
            most_inelastic_category = category_elasticity['correlation'].idxmax() if not category_elasticity.empty else 'Unknown'
            most_inelastic_value = category_elasticity['correlation'].max() if not category_elasticity.empty else 0
            dominant_price_segment = price_segment_perf['final_amount_inr'].idxmax()
            dominant_segment_revenue_share = (price_segment_perf['final_amount_inr'].max() / 
                                            price_segment_perf['final_amount_inr'].sum()) * 100
            
            insight = (f"Q10 Insight: Overall price-demand correlation: {price_demand_corr:.3f}. "
                    f"{most_elastic_category} most elastic ({most_elastic_value:.3f}), "
                    f"{most_inelastic_category} most inelastic ({most_inelastic_value:.3f}). "
                    f"{dominant_price_segment} segment dominates ({dominant_segment_revenue_share:.1f}% revenue).")
            
            self.insights.append(insight)
            logger.info(f"✅ Price vs demand analysis completed. Correlation: {price_demand_corr:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Price vs demand analysis failed: {e}")
            import traceback
            logger.error(f"🔍 Detailed error: {traceback.format_exc()}")
            self.insights.append("Q10 Insight: Price vs demand analysis failed due to data issues.")
    
    def q11_delivery_performance(self):
        """Question 11: Delivery performance analysis across cities and customer tiers"""
        logger.info("EDA Question 11: Delivery performance analysis")
        
        try:
            # Clean delivery days data
            self.df['delivery_days_clean'] = pd.to_numeric(self.df['delivery_days'], errors='coerce')
            # Remove outliers (delivery days > 30 considered unrealistic)
            self.df['delivery_days_clean'] = self.df['delivery_days_clean'].clip(upper=30)
            
            # Calculate delivery performance metrics
            delivery_metrics = self.df.groupby('customer_city').agg({
                'delivery_days_clean': ['mean', 'median', 'count'],
                'customer_rating': 'mean',
                'final_amount_inr': 'sum'
            }).round(2)
            
            delivery_metrics.columns = ['avg_delivery_days', 'median_delivery_days', 'order_count', 'avg_rating', 'total_revenue']
            
            # Filter cities with sufficient data
            delivery_metrics = delivery_metrics[delivery_metrics['order_count'] > 100]
            
            # Calculate on-time performance (assuming <= 7 days is on-time)
            self.df['on_time_delivery'] = self.df['delivery_days_clean'] <= 7
            on_time_by_city = self.df.groupby('customer_city')['on_time_delivery'].mean() * 100
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Average delivery days by city (top 15)
            top_cities_delivery = delivery_metrics.nlargest(15, 'order_count')['avg_delivery_days'].sort_values()
            bars1 = ax1.barh(range(len(top_cities_delivery)), top_cities_delivery.values, 
                           color='#1f77b4', alpha=0.7)
            ax1.set_yticks(range(len(top_cities_delivery)))
            ax1.set_yticklabels(top_cities_delivery.index)
            ax1.set_title('Average Delivery Days by City (Top 15)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Average Delivery Days')
            ax1.axvline(x=7, color='red', linestyle='--', alpha=0.7, label='7-day target')
            ax1.legend()
            
            # Plot 2: On-time delivery performance
            on_time_top_cities = on_time_by_city.reindex(top_cities_delivery.index).sort_values(ascending=False)
            bars2 = ax2.bar(range(len(on_time_top_cities)), on_time_top_cities.values,
                          color='#2ca02c', alpha=0.7)
            ax2.set_title('On-time Delivery Rate by City (%)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('City')
            ax2.set_ylabel('On-time Delivery Rate (%)')
            ax2.set_xticks(range(len(on_time_top_cities)))
            ax2.set_xticklabels(on_time_top_cities.index, rotation=45, ha='right')
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% target')
            ax2.legend()
            
            # Plot 3: Delivery performance by city tier
            tier_delivery = self.df.groupby('city_tier').agg({
                'delivery_days_clean': 'mean',
                'on_time_delivery': 'mean'
            }).round(2)
            tier_delivery['on_time_delivery'] *= 100
            
            x = np.arange(len(tier_delivery))
            width = 0.35
            
            ax3.bar(x - width/2, tier_delivery['delivery_days_clean'], width, 
                   label='Avg Delivery Days', color='#ff7f0e', alpha=0.7)
            ax3_twin = ax3.twinx()
            ax3_twin.bar(x + width/2, tier_delivery['on_time_delivery'], width,
                        label='On-time Rate (%)', color='#17becf', alpha=0.7)
            
            ax3.set_title('Delivery Performance by City Tier', fontsize=14, fontweight='bold')
            ax3.set_xlabel('City Tier')
            ax3.set_ylabel('Average Delivery Days')
            ax3_twin.set_ylabel('On-time Delivery Rate (%)')
            ax3.set_xticks(x)
            ax3.set_xticklabels(tier_delivery.index)
            ax3.legend(loc='upper left')
            ax3_twin.legend(loc='upper right')
            
            # Plot 4: Correlation between delivery speed and customer rating
            scatter = ax4.scatter(self.df['delivery_days_clean'], self.df['customer_rating'],
                                alpha=0.5, s=20, c=self.df['final_amount_inr'], cmap='viridis')
            ax4.set_xlabel('Delivery Days')
            ax4.set_ylabel('Customer Rating')
            ax4.set_title('Delivery Speed vs Customer Rating\n(Color: Order Value)', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax4, label='Order Value (INR)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'q11_delivery_performance.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate insights
            fastest_city = top_cities_delivery.idxmin()
            fastest_delivery = top_cities_delivery.min()
            slowest_city = top_cities_delivery.idxmax()
            slowest_delivery = top_cities_delivery.max()
            best_on_time_city = on_time_top_cities.idxmax()
            best_on_time_rate = on_time_top_cities.max()
            
            # Calculate correlation
            correlation = self.df[['delivery_days_clean', 'customer_rating']].corr().iloc[0,1]
            
            insight = (f"Q11 Insight: {fastest_city} has fastest delivery ({fastest_delivery:.1f} days), "
                      f"{slowest_city} slowest ({slowest_delivery:.1f} days). "
                      f"{best_on_time_city} leads on-time delivery ({best_on_time_rate:.1f}%). "
                      f"Delivery-rating correlation: {correlation:.3f}.")
            
            self.insights.append(insight)
            
        except Exception as e:
            logger.error(f"Delivery performance analysis failed: {e}")
            self.insights.append("Q11 Insight: Delivery performance analysis failed due to data issues.")

    def q12_return_patterns(self):
        """Question 12: Return patterns and customer satisfaction analysis using subcategories"""
        logger.info("EDA Question 12: Return patterns analysis")
        
        try:
            # Analyze return patterns
            return_analysis = self.df.groupby('return_status').agg({
                'transaction_id': 'count',
                'final_amount_inr': 'sum',
                'customer_rating': 'mean',
                'discount_percent': 'mean'
            }).round(2)
            
            return_analysis['percentage'] = (return_analysis['transaction_id'] / return_analysis['transaction_id'].sum()) * 100
            
            # Subcategory-wise return analysis
            subcategory_returns = self.df.groupby(['subcategory', 'return_status']).size().unstack(fill_value=0)
            subcategory_returns['return_rate'] = (subcategory_returns.get('Returned', 0) / subcategory_returns.sum(axis=1)) * 100
            
            # Price vs return rate analysis
            self.df['price_segment'] = pd.cut(self.df['final_amount_inr'], 
                                            bins=[0, 1000, 5000, 10000, 50000, float('inf')],
                                            labels=['<1K', '1K-5K', '5K-10K', '10K-50K', '>50K'])
            
            price_segment_returns = self.df.groupby(['price_segment', 'return_status']).size().unstack(fill_value=0)
            price_segment_returns['return_rate'] = (price_segment_returns.get('Returned', 0) / price_segment_returns.sum(axis=1)) * 100
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Return status distribution
            colors1 = ['#2ca02c', '#d62728', '#ff7f0e']  # Green for Not Returned, Red for Returned
            wedges1, texts1, autotexts1 = ax1.pie(return_analysis['transaction_id'], 
                                                labels=return_analysis.index,
                                                autopct='%1.1f%%', colors=colors1, startangle=90)
            ax1.set_title('Return Status Distribution', fontsize=14, fontweight='bold')
            
            # Plot 2: Subcategory-wise return rates
            top_subcategories_returns = subcategory_returns.nlargest(10, 'return_rate')['return_rate'].sort_values()
            bars2 = ax2.barh(range(len(top_subcategories_returns)), top_subcategories_returns.values,
                        color='#d62728', alpha=0.7)
            ax2.set_yticks(range(len(top_subcategories_returns)))
            ax2.set_yticklabels(top_subcategories_returns.index)
            ax2.set_title('Return Rates by Subcategory (Top 10)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Return Rate (%)')
            ax2.axvline(x=subcategory_returns['return_rate'].mean(), color='red', linestyle='--', 
                    label=f'Avg: {subcategory_returns["return_rate"].mean():.1f}%')
            ax2.legend()
            
            # Plot 3: Price segment vs return rate
            bars3 = ax3.bar(range(len(price_segment_returns)), price_segment_returns['return_rate'].values,
                        color='#ff7f0e', alpha=0.7)
            ax3.set_title('Return Rates by Price Segment', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Price Segment (INR)')
            ax3.set_ylabel('Return Rate (%)')
            ax3.set_xticks(range(len(price_segment_returns)))
            ax3.set_xticklabels(price_segment_returns.index, rotation=45)
            
            # Plot 4: Rating distribution for returned vs non-returned products
            returned_ratings = self.df[self.df['return_status'] == 'Returned']['customer_rating'].dropna()
            non_returned_ratings = self.df[self.df['return_status'] == 'Not Returned']['customer_rating'].dropna()
            
            ax4.hist([non_returned_ratings, returned_ratings], bins=10, alpha=0.7, 
                    label=['Not Returned', 'Returned'], color=['#2ca02c', '#d62728'])
            ax4.set_title('Customer Rating Distribution: Returned vs Non-Returned', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Customer Rating')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'q12_return_patterns.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate insights
            overall_return_rate = return_analysis.loc['Returned', 'percentage'] if 'Returned' in return_analysis.index else 0
            highest_return_subcategory = subcategory_returns['return_rate'].idxmax()
            highest_return_rate = subcategory_returns['return_rate'].max()
            highest_return_price_segment = price_segment_returns['return_rate'].idxmax()
            highest_return_price_rate = price_segment_returns['return_rate'].max()
            
            avg_rating_returned = returned_ratings.mean() if len(returned_ratings) > 0 else 0
            avg_rating_non_returned = non_returned_ratings.mean() if len(non_returned_ratings) > 0 else 0
            
            insight = (f"Q12 Insight: Overall return rate: {overall_return_rate:.1f}%. "
                    f"{highest_return_subcategory} has highest return rate ({highest_return_rate:.1f}%). "
                    f"{highest_return_price_segment} price segment returns most ({highest_return_price_rate:.1f}%). "
                    f"Returned items rated {avg_rating_returned:.1f} vs non-returned {avg_rating_non_returned:.1f}.")
            
            self.insights.append(insight)
            
        except Exception as e:
            logger.error(f"Return patterns analysis failed: {e}")
            self.insights.append("Q12 Insight: Return patterns analysis failed due to data issues.")

    def q13_brand_performance(self):
        """Question 13: Brand performance and market share evolution with subcategory context"""
        logger.info("EDA Question 13: Brand performance analysis")
        
        try:
            # Calculate brand performance metrics
            brand_performance = self.df.groupby('brand').agg({
                'final_amount_inr': ['sum', 'mean', 'count'],
                'quantity': 'sum',
                'customer_rating': 'mean',
                'discount_percent': 'mean'
            }).round(2)
            
            brand_performance.columns = ['total_revenue', 'avg_order_value', 'orders', 'total_quantity', 'avg_rating', 'avg_discount']
            
            # Filter brands with sufficient data
            brand_performance = brand_performance[brand_performance['orders'] > 100]
            
            # Calculate market share
            total_revenue = brand_performance['total_revenue'].sum()
            brand_performance['market_share'] = (brand_performance['total_revenue'] / total_revenue) * 100
            
            # Brand evolution over time
            brand_yearly = self.df.groupby(['order_year', 'brand'])['final_amount_inr'].sum().unstack(fill_value=0)
            top_brands = brand_performance.nlargest(10, 'market_share').index
            brand_yearly_top = brand_yearly[top_brands]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
            
            # Plot 1: Top brands by market share
            top_brands_share = brand_performance.nlargest(10, 'market_share')['market_share'].sort_values()
            bars1 = ax1.barh(range(len(top_brands_share)), top_brands_share.values,
                        color='#1f77b4', alpha=0.7)
            ax1.set_yticks(range(len(top_brands_share)))
            ax1.set_yticklabels(top_brands_share.index)
            ax1.set_title('Top 10 Brands by Market Share', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Market Share (%)')
            
            # Add value labels
            for i, bar in enumerate(bars1):
                width = bar.get_width()
                ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                        ha='left', va='center', fontsize=9)
            
            # Plot 2: Brand revenue evolution (top 5)
            top_5_brands = brand_performance.nlargest(5, 'market_share').index
            brand_yearly_top5 = brand_yearly[top_5_brands]
            
            for brand in top_5_brands:
                ax2.plot(brand_yearly_top5.index, brand_yearly_top5[brand], marker='o', linewidth=2, label=brand)
            
            ax2.set_title('Revenue Evolution of Top 5 Brands', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Revenue (INR)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Brand positioning (Rating vs Market Share)
            scatter = ax3.scatter(brand_performance['avg_rating'], brand_performance['market_share'],
                                s=brand_performance['total_revenue']/1000, alpha=0.6,
                                c=brand_performance['avg_discount'], cmap='coolwarm')
            ax3.set_xlabel('Average Rating')
            ax3.set_ylabel('Market Share (%)')
            ax3.set_title('Brand Positioning: Rating vs Market Share\n(Size: Revenue, Color: Discount %)', 
                        fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax3, label='Average Discount (%)')
            
            # Add brand labels for top brands
            for brand in top_brands_share.index:
                row = brand_performance.loc[brand]
                ax3.annotate(brand, (row['avg_rating'], row['market_share']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Plot 4: Subcategory-wise brand dominance
            subcategory_brand_revenue = self.df.groupby(['subcategory', 'brand'])['final_amount_inr'].sum().reset_index()
            top_brands_by_subcategory = subcategory_brand_revenue.loc[subcategory_brand_revenue.groupby('subcategory')['final_amount_inr'].idxmax()]
            
            bars4 = ax4.bar(range(len(top_brands_by_subcategory)), top_brands_by_subcategory['final_amount_inr'].values,
                        color='#2ca02c', alpha=0.7)
            ax4.set_title('Leading Brands by Subcategory', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Subcategory')
            ax4.set_ylabel('Revenue (INR)')
            ax4.set_xticks(range(len(top_brands_by_subcategory)))
            ax4.set_xticklabels(top_brands_by_subcategory['subcategory'], rotation=45, ha='right')
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'q13_brand_performance.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate insights
            top_brand = top_brands_share.index[-1]  # Last one is largest due to sorting
            top_brand_share = top_brands_share.iloc[-1]
            fastest_growing_brand = brand_yearly_top.pct_change().mean().idxmax()
            highest_rated_brand = brand_performance['avg_rating'].idxmax()
            highest_rating = brand_performance['avg_rating'].max()
            
            # Subcategory insights
            dominant_subcategory_brand = top_brands_by_subcategory.iloc[0]
            dominant_subcategory = dominant_subcategory_brand['subcategory']
            dominant_brand = dominant_subcategory_brand['brand']
            
            insight = (f"Q13 Insight: {top_brand} leads with {top_brand_share:.1f}% market share. "
                    f"{fastest_growing_brand} shows strongest growth. "
                    f"{highest_rated_brand} has highest rating ({highest_rating:.1f}/5). "
                    f"{dominant_brand} dominates {dominant_subcategory} subcategory.")
            
            self.insights.append(insight)
            
        except Exception as e:
            logger.error(f"Brand performance analysis failed: {e}")
            self.insights.append("Q13 Insight: Brand performance analysis failed due to data issues.")

    def q14_customer_lifetime_value(self):
        """Question 14: Customer Lifetime Value (CLV) analysis using cohort analysis"""
        logger.info("EDA Question 14: Customer Lifetime Value analysis")
        
        try:
            # Calculate basic CLV metrics
            customer_metrics = self.df.groupby('customer_id').agg({
                'order_date': ['min', 'max', 'count'],
                'final_amount_inr': 'sum',
                'customer_city': 'first',
                'is_prime_member': 'first'
            }).round(2)
            
            customer_metrics.columns = ['first_purchase', 'last_purchase', 'total_orders', 'total_spent', 'city', 'is_prime']
            
            # Calculate customer lifetime in days
            customer_metrics['lifetime_days'] = (customer_metrics['last_purchase'] - customer_metrics['first_purchase']).dt.days
            customer_metrics['avg_order_value'] = customer_metrics['total_spent'] / customer_metrics['total_orders']
            
            # Simple CLV calculation (total spent per customer)
            customer_metrics['clv'] = customer_metrics['total_spent']
            
            # Cohort analysis for CLV
            self.df['cohort_year'] = self.df.groupby('customer_id')['order_date'].transform('min').dt.year
            cohort_clv = self.df.groupby('cohort_year').agg({
                'customer_id': 'nunique',
                'final_amount_inr': 'sum'
            })
            cohort_clv['avg_clv'] = cohort_clv['final_amount_inr'] / cohort_clv['customer_id']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: CLV distribution
            clv_data = customer_metrics[customer_metrics['clv'] > 0]['clv']
            ax1.hist(clv_data, bins=50, color='#1f77b4', alpha=0.7, edgecolor='black')
            ax1.set_title('Customer Lifetime Value Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('CLV (INR)')
            ax1.set_ylabel('Number of Customers')
            ax1.set_xlim(0, clv_data.quantile(0.95))  # Remove extreme outliers for better visualization
            
            # Plot 2: Cohort-wise average CLV
            bars2 = ax2.bar(cohort_clv.index, cohort_clv['avg_clv'].values, color='#ff7f0e', alpha=0.7)
            ax2.set_title('Average CLV by Acquisition Cohort', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Acquisition Year')
            ax2.set_ylabel('Average CLV (INR)')
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'₹{height:,.0f}', ha='center', va='bottom')
            
            # Plot 3: CLV by city tier
            customer_metrics_with_tier = customer_metrics.copy()
            customer_metrics_with_tier['city_tier'] = customer_metrics_with_tier['city'].apply(
                lambda x: self.classify_city_tier(x) if hasattr(self, 'classify_city_tier') else 'Unknown'
            )
            
            clv_by_tier = customer_metrics_with_tier.groupby('city_tier')['clv'].mean()
            bars3 = ax3.bar(clv_by_tier.index, clv_by_tier.values, color='#2ca02c', alpha=0.7)
            ax3.set_title('Average CLV by City Tier', fontsize=14, fontweight='bold')
            ax3.set_xlabel('City Tier')
            ax3.set_ylabel('Average CLV (INR)')
            
            # Plot 4: CLV by Prime membership
            clv_by_prime = customer_metrics.groupby('is_prime')['clv'].mean()
            bars4 = ax4.bar(['Non-Prime', 'Prime'], clv_by_prime.values, color=['#ff7f0e', '#1f77b4'], alpha=0.7)
            ax4.set_title('Average CLV: Prime vs Non-Prime', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Average CLV (INR)')
            
            # Add value labels
            for bar in bars4:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'₹{height:,.0f}', ha='center', va='bottom')
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'q14_customer_lifetime_value.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate insights
            avg_clv = customer_metrics['clv'].mean()
            median_clv = customer_metrics['clv'].median()
            top_10_percent_clv = customer_metrics['clv'].quantile(0.9)
            prime_clv = clv_by_prime[True] if True in clv_by_prime.index else 0
            non_prime_clv = clv_by_prime[False] if False in clv_by_prime.index else 0
            prime_clv_advantage = ((prime_clv - non_prime_clv) / non_prime_clv) * 100 if non_prime_clv > 0 else 0
            
            highest_clv_tier = clv_by_tier.idxmax() if len(clv_by_tier) > 0 else 'Unknown'
            highest_clv_tier_value = clv_by_tier.max() if len(clv_by_tier) > 0 else 0
            
            insight = (f"Q14 Insight: Average CLV: ₹{avg_clv:,.0f}, Median: ₹{median_clv:,.0f}. "
                      f"Top 10% customers worth ₹{top_10_percent_clv:,.0f}+. "
                      f"Prime members CLV {prime_clv_advantage:+.1f}% higher. "
                      f"{highest_clv_tier} cities have highest CLV (₹{highest_clv_tier_value:,.0f}).")
            
            self.insights.append(insight)
            
        except Exception as e:
            logger.error(f"CLV analysis failed: {e}")
            self.insights.append("Q14 Insight: Customer Lifetime Value analysis failed due to data issues.")

    def q15_discount_effectiveness(self):
        """Question 15: Discount and promotional effectiveness analysis"""
        logger.info("EDA Question 15: Discount effectiveness analysis")
        
        try:
            # Create discount segments
            self.df['discount_segment'] = pd.cut(self.df['discount_percent'], 
                                               bins=[-1, 0, 10, 20, 30, 40, 50, 100],
                                               labels=['0%', '1-10%', '11-20%', '21-30%', '31-40%', '41-50%', '50%+'])
            
            # Analyze discount effectiveness
            discount_analysis = self.df.groupby('discount_segment').agg({
                'transaction_id': 'count',
                'final_amount_inr': 'sum',
                'quantity': 'sum',
                'customer_rating': 'mean'
            }).round(2)
            
            discount_analysis['avg_order_value'] = discount_analysis['final_amount_inr'] / discount_analysis['transaction_id']
            discount_analysis['conversion_rate'] = (discount_analysis['transaction_id'] / discount_analysis['transaction_id'].sum()) * 100
            
            # Category-wise discount analysis
            category_discount = self.df.groupby(['category', 'discount_segment'])['final_amount_inr'].sum().unstack(fill_value=0)
            
            # Time-based discount analysis
            yearly_discount_trend = self.df.groupby('order_year')['discount_percent'].mean()
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Discount segment performance
            x_pos = np.arange(len(discount_analysis))
            width = 0.35
            
            bars1a = ax1.bar(x_pos - width/2, discount_analysis['transaction_id'], width,
                           label='Number of Orders', color='#1f77b4', alpha=0.7)
            ax1_twin = ax1.twinx()
            bars1b = ax1_twin.bar(x_pos + width/2, discount_analysis['avg_order_value'], width,
                                label='Avg Order Value', color='#ff7f0e', alpha=0.7)
            
            ax1.set_title('Discount Segment Performance', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Discount Segment')
            ax1.set_ylabel('Number of Orders')
            ax1_twin.set_ylabel('Average Order Value (INR)')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(discount_analysis.index, rotation=45)
            ax1.legend(loc='upper left')
            ax1_twin.legend(loc='upper right')
            
            # Plot 2: Discount trend over years
            ax2.plot(yearly_discount_trend.index, yearly_discount_trend.values, 
                    marker='o', linewidth=2, color='#2ca02c')
            ax2.set_title('Average Discount Trend Over Years', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Average Discount (%)')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Category-wise discount effectiveness (heatmap)
            category_discount_pct = category_discount.div(category_discount.sum(axis=1), axis=0) * 100
            sns.heatmap(category_discount_pct, annot=True, fmt='.1f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Revenue Share (%)'}, ax=ax3)
            ax3.set_title('Discount Strategy by Category\n(Revenue Distribution)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Discount Segment')
            ax3.set_ylabel('Category')
            
            # Plot 4: Discount vs Rating correlation
            scatter = ax4.scatter(self.df['discount_percent'], self.df['customer_rating'],
                                alpha=0.5, s=20, c=self.df['final_amount_inr'], cmap='viridis')
            ax4.set_xlabel('Discount Percentage')
            ax4.set_ylabel('Customer Rating')
            ax4.set_title('Discount vs Customer Rating\n(Color: Order Value)', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax4, label='Order Value (INR)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'q15_discount_effectiveness.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate insights
            most_popular_discount = discount_analysis['transaction_id'].idxmax()
            highest_aov_discount = discount_analysis['avg_order_value'].idxmax()
            highest_aov_value = discount_analysis['avg_order_value'].max()
            avg_discount = self.df['discount_percent'].mean()
            
            # Calculate discount-rating correlation
            discount_rating_corr = self.df[['discount_percent', 'customer_rating']].corr().iloc[0,1]
            
            insight = (f"Q15 Insight: {most_popular_discount} discount most common. "
                      f"Highest AOV at {highest_aov_discount} discount (₹{highest_aov_value:,.0f}). "
                      f"Average discount: {avg_discount:.1f}%. "
                      f"Discount-rating correlation: {discount_rating_corr:.3f}.")
            
            self.insights.append(insight)
            
        except Exception as e:
            logger.error(f"Discount effectiveness analysis failed: {e}")
            self.insights.append("Q15 Insight: Discount effectiveness analysis failed due to data issues.")

    def q16_rating_impact_analysis(self):
        """Question 16: Product rating patterns and their impact on sales using subcategories"""
        logger.info("EDA Question 16: Rating impact analysis")
        
        try:
            # Analyze rating distribution and impact
            rating_analysis = self.df.groupby('product_rating').agg({
                'transaction_id': 'count',
                'final_amount_inr': 'sum',
                'quantity': 'sum',
                'discount_percent': 'mean'
            }).round(2)
            
            rating_analysis['avg_order_value'] = rating_analysis['final_amount_inr'] / rating_analysis['transaction_id']
            
            # Rating distribution by subcategory
            rating_by_subcategory = self.df.groupby(['subcategory', 'product_rating'])['transaction_id'].count().unstack(fill_value=0)
            rating_by_subcategory_pct = rating_by_subcategory.div(rating_by_subcategory.sum(axis=1), axis=0) * 100
            
            # Price vs Rating analysis
            self.df['price_group'] = pd.cut(self.df['final_amount_inr'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            price_rating = self.df.groupby('price_group')['product_rating'].mean()
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Rating distribution
            bars1 = ax1.bar(rating_analysis.index, rating_analysis['transaction_id'].values,
                        color='#1f77b4', alpha=0.7)
            ax1.set_title('Product Rating Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Product Rating')
            ax1.set_ylabel('Number of Transactions')
            ax1.set_xticks(rating_analysis.index)
            
            # Plot 2: Rating vs Sales performance
            x_pos = np.arange(len(rating_analysis))
            width = 0.35
            
            bars2a = ax2.bar(x_pos - width/2, rating_analysis['transaction_id'], width,
                        label='Transaction Count', color='#1f77b4', alpha=0.7)
            ax2_twin = ax2.twinx()
            bars2b = ax2_twin.bar(x_pos + width/2, rating_analysis['avg_order_value'], width,
                                label='Avg Order Value', color='#ff7f0e', alpha=0.7)
            
            ax2.set_title('Rating vs Sales Performance', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Product Rating')
            ax2.set_ylabel('Transaction Count')
            ax2_twin.set_ylabel('Average Order Value (INR)')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(rating_analysis.index)
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
            
            # Plot 3: Rating distribution by subcategory (heatmap)
            # Limit to top subcategories for better visualization
            top_subcategories = rating_by_subcategory_pct.nlargest(10, 5.0) if 5.0 in rating_by_subcategory_pct.columns else rating_by_subcategory_pct.head(10)
            
            sns.heatmap(top_subcategories, annot=True, fmt='.1f', cmap='YlOrRd', 
                    cbar_kws={'label': 'Percentage (%)'}, ax=ax3)
            ax3.set_title('Rating Distribution by Subcategory (Top 10)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Product Rating')
            ax3.set_ylabel('Subcategory')
            
            # Plot 4: Price vs Average Rating
            bars4 = ax4.bar(price_rating.index, price_rating.values, color='#2ca02c', alpha=0.7)
            ax4.set_title('Average Rating by Price Group', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Price Group')
            ax4.set_ylabel('Average Rating')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'q16_rating_impact_analysis.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate insights
            most_common_rating = rating_analysis['transaction_id'].idxmax()
            highest_rated_subcategory = rating_by_subcategory_pct[5.0].idxmax() if 5.0 in rating_by_subcategory_pct.columns else 'Unknown'
            highest_5star_rate = rating_by_subcategory_pct[5.0].max() if 5.0 in rating_by_subcategory_pct.columns else 0
            avg_rating = self.df['product_rating'].mean()
            
            # Calculate correlation between rating and sales
            rating_sales_corr = self.df[['product_rating', 'final_amount_inr']].corr().iloc[0,1]
            
            insight = (f"Q16 Insight: Most common rating: {most_common_rating}. "
                    f"Average product rating: {avg_rating:.2f}. "
                    f"{highest_rated_subcategory} has highest 5-star rate ({highest_5star_rate:.1f}%). "
                    f"Rating-sales correlation: {rating_sales_corr:.3f}.")
            
            self.insights.append(insight)
            
        except Exception as e:
            logger.error(f"Rating impact analysis failed: {e}")
            self.insights.append("Q16 Insight: Rating impact analysis failed due to data issues.")

    def q17_prime_member_analysis(self):
        """Question 17: Prime membership impact analysis"""
        logger.info("EDA Question 17: Prime membership impact analysis")
        
        try:
            # Use correct column name - check which prime column exists
            prime_columns = [col for col in self.df.columns if 'prime' in col.lower()]
            if not prime_columns:
                raise KeyError("No prime-related columns found in dataset")
            
            prime_column = prime_columns[0]  # Use the first prime-related column found
            logger.info(f"Using prime column: {prime_column}")
            
            # Prime vs Non-Prime comparison
            prime_comparison = self.df.groupby(prime_column).agg({
                'transaction_id': 'count',
                'final_amount_inr': ['sum', 'mean'],
                'quantity': 'mean',
                'customer_rating': 'mean',
                'discount_percent': 'mean'
            }).round(2)
            
            # Flatten column names
            prime_comparison.columns = ['order_count', 'total_revenue', 'avg_order_value', 'avg_quantity', 'avg_rating', 'avg_discount']
            prime_comparison['percentage_of_orders'] = (prime_comparison['order_count'] / prime_comparison['order_count'].sum()) * 100
            
            # Prime member behavior over time
            prime_trend = self.df.groupby(['order_year', prime_column])['transaction_id'].count().unstack(fill_value=0)
            prime_trend_pct = prime_trend.div(prime_trend.sum(axis=1), axis=0) * 100
            
            # Category preference for Prime vs Non-Prime
            prime_category = self.df.groupby(['category', prime_column])['final_amount_inr'].sum().unstack(fill_value=0)
            prime_category_pct = prime_category.div(prime_category.sum(axis=1), axis=0) * 100
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Prime vs Non-Prime comparison
            metrics = ['avg_order_value', 'avg_quantity', 'avg_rating', 'avg_discount']
            
            # Get values for prime and non-prime
            prime_values = []
            non_prime_values = []
            
            for metric in metrics:
                if True in prime_comparison.index:
                    prime_values.append(prime_comparison.loc[True, metric])
                else:
                    prime_values.append(0)
                
                if False in prime_comparison.index:
                    non_prime_values.append(prime_comparison.loc[False, metric])
                else:
                    non_prime_values.append(0)
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1a = ax1.bar(x - width/2, prime_values, width, label='Prime', color='#ff9900', alpha=0.7)
            bars1b = ax1.bar(x + width/2, non_prime_values, width, label='Non-Prime', color='#1f77b4', alpha=0.7)
            
            ax1.set_title('Prime vs Non-Prime: Key Metrics Comparison', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Values')
            ax1.set_xticks(x)
            ax1.set_xticklabels(['Avg Order\nValue', 'Avg Quantity', 'Avg Rating', 'Avg Discount'])
            ax1.legend()
            
            # Plot 2: Prime membership trend
            if True in prime_trend_pct.columns:
                ax2.plot(prime_trend_pct.index, prime_trend_pct[True], marker='o', linewidth=2, color='#ff9900', label='Prime')
            if False in prime_trend_pct.columns:
                ax2.plot(prime_trend_pct.index, prime_trend_pct[False], marker='o', linewidth=2, color='#1f77b4', label='Non-Prime')
            ax2.set_title('Prime Membership Trend Over Years', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Percentage of Orders (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Category preference heatmap
            if not prime_category_pct.empty:
                sns.heatmap(prime_category_pct, annot=True, fmt='.1f', cmap='YlOrRd', 
                        cbar_kws={'label': 'Revenue Share (%)'}, ax=ax3)
                ax3.set_title('Prime vs Non-Prime: Category Preferences', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Prime Membership')
                ax3.set_ylabel('Category')
            else:
                ax3.text(0.5, 0.5, 'Category preference data not available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Category Preferences (Data Not Available)', fontsize=14, fontweight='bold')
            
            # Plot 4: Order frequency comparison
            customer_order_freq = self.df.groupby(['customer_id', prime_column])['transaction_id'].count().reset_index()
            
            prime_order_freq = []
            non_prime_order_freq = []
            
            if True in customer_order_freq[prime_column].values:
                prime_order_freq = customer_order_freq[customer_order_freq[prime_column] == True]['transaction_id'].tolist()
            if False in customer_order_freq[prime_column].values:
                non_prime_order_freq = customer_order_freq[customer_order_freq[prime_column] == False]['transaction_id'].tolist()
            
            if prime_order_freq and non_prime_order_freq:
                ax4.boxplot([non_prime_order_freq, prime_order_freq], 
                        labels=['Non-Prime', 'Prime'], patch_artist=True,
                        boxprops=dict(facecolor='#1f77b4', alpha=0.7),
                        medianprops=dict(color='red'))
                ax4.set_title('Order Frequency: Prime vs Non-Prime', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Number of Orders per Customer')
            else:
                ax4.text(0.5, 0.5, 'Order frequency data not available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Order Frequency (Data Not Available)', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'q17_prime_member_analysis.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate insights
            prime_percentage = prime_comparison.loc[True, 'percentage_of_orders'] if True in prime_comparison.index else 0
            prime_aov_advantage = 0
            if True in prime_comparison.index and False in prime_comparison.index:
                prime_aov = prime_comparison.loc[True, 'avg_order_value']
                non_prime_aov = prime_comparison.loc[False, 'avg_order_value']
                prime_aov_advantage = ((prime_aov - non_prime_aov) / non_prime_aov) * 100 if non_prime_aov > 0 else 0
            
            prime_growth = 0
            if True in prime_trend_pct.columns and len(prime_trend_pct) > 1:
                prime_growth = ((prime_trend_pct[True].iloc[-1] - prime_trend_pct[True].iloc[0]) / 
                            prime_trend_pct[True].iloc[0]) * 100
            
            insight = (f"Q17 Insight: Prime members: {prime_percentage:.1f}% of orders. "
                    f"Prime AOV {prime_aov_advantage:+.1f}% higher. "
                    f"Prime membership grew {prime_growth:+.1f}% over period.")
            
            self.insights.append(insight)
            
        except Exception as e:
            logger.error(f"Prime member analysis failed: {e}")
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
            self.insights.append("Q17 Insight: Prime membership analysis failed due to data issues.")

    def q18_customer_segmentation(self):
        """Question 18: Customer segmentation using RFM analysis"""
        logger.info("EDA Question 18: Customer segmentation analysis")
        
        try:
            # RFM Analysis
            current_date = self.df['order_date'].max()
            
            rfm = self.df.groupby('customer_id').agg({
                'order_date': lambda x: (current_date - x.max()).days,  # Recency
                'transaction_id': 'count',  # Frequency
                'final_amount_inr': 'sum'   # Monetary
            }).round(2)
            
            rfm.columns = ['recency', 'frequency', 'monetary']
            
            # Create RFM scores
            rfm['r_score'] = pd.qcut(rfm['recency'], 4, labels=[4, 3, 2, 1])
            rfm['f_score'] = pd.qcut(rfm['frequency'], 4, labels=[1, 2, 3, 4])
            rfm['m_score'] = pd.qcut(rfm['monetary'], 4, labels=[1, 2, 3, 4])
            
            rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
            
            # Create segments
            def segment_customer(row):
                if row['r_score'] == 4 and row['f_score'] == 4 and row['m_score'] == 4:
                    return 'Champions'
                elif row['r_score'] in [3,4] and row['f_score'] in [3,4] and row['m_score'] in [3,4]:
                    return 'Loyal Customers'
                elif row['r_score'] in [3,4] and row['f_score'] in [1,2]:
                    return 'New Customers'
                elif row['r_score'] in [2,3] and row['f_score'] in [2,3]:
                    return 'Potential Loyalists'
                elif row['r_score'] in [1,2] and row['f_score'] in [3,4]:
                    return 'At Risk'
                elif row['r_score'] == 1 and row['f_score'] in [1,2]:
                    return 'Lost Customers'
                else:
                    return 'Others'
            
            rfm['segment'] = rfm.apply(segment_customer, axis=1)
            
            # Segment analysis
            segment_analysis = rfm.groupby('segment').agg({
                'customer_id': 'count',
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': 'mean'
            }).round(2)
            
            segment_analysis['percentage'] = (segment_analysis['customer_id'] / segment_analysis['customer_id'].sum()) * 100
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Customer segments distribution
            colors1 = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
            wedges1, texts1, autotexts1 = ax1.pie(segment_analysis['customer_id'], 
                                                 labels=segment_analysis.index,
                                                 autopct='%1.1f%%', colors=colors1, startangle=90)
            ax1.set_title('Customer Segments Distribution', fontsize=14, fontweight='bold')
            
            # Plot 2: RFM distribution by segment
            segments_plot = segment_analysis.nlargest(6, 'customer_id').index
            rfm_metrics = ['recency', 'frequency', 'monetary']
            
            x = np.arange(len(segments_plot))
            width = 0.25
            
            for i, metric in enumerate(rfm_metrics):
                values = [segment_analysis.loc[segment, metric] for segment in segments_plot]
                ax2.bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.7)
            
            ax2.set_title('RFM Metrics by Customer Segment', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Customer Segment')
            ax2.set_ylabel('Values')
            ax2.set_xticks(x + width)
            ax2.set_xticklabels(segments_plot, rotation=45, ha='right')
            ax2.legend()
            
            # Plot 3: Monetary vs Frequency scatter plot
            scatter = ax3.scatter(rfm['frequency'], rfm['monetary'], 
                                c=rfm['recency'], alpha=0.6, cmap='viridis', s=30)
            ax3.set_xlabel('Frequency')
            ax3.set_ylabel('Monetary Value (INR)')
            ax3.set_title('Customer Segmentation: Frequency vs Monetary\n(Color: Recency)', 
                         fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax3, label='Recency (Days)')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Segment-wise customer value
            segment_value = segment_analysis[['monetary', 'customer_id']].copy()
            segment_value['total_value'] = segment_value['monetary'] * segment_value['customer_id']
            segment_value = segment_value.nlargest(8, 'total_value')
            
            bars4 = ax4.bar(range(len(segment_value)), segment_value['total_value'].values,
                          color='#ff7f0e', alpha=0.7)
            ax4.set_title('Total Customer Value by Segment', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Customer Segment')
            ax4.set_ylabel('Total Value (INR)')
            ax4.set_xticks(range(len(segment_value)))
            ax4.set_xticklabels(segment_value.index, rotation=45, ha='right')
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'q18_customer_segmentation.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate insights
            largest_segment = segment_analysis['customer_id'].idxmax()
            largest_segment_pct = segment_analysis.loc[largest_segment, 'percentage']
            highest_value_segment = segment_analysis['monetary'].idxmax()
            highest_avg_value = segment_analysis.loc[highest_value_segment, 'monetary']
            champions_pct = segment_analysis.loc['Champions', 'percentage'] if 'Champions' in segment_analysis.index else 0
            
            insight = (f"Q18 Insight: {largest_segment} largest segment ({largest_segment_pct:.1f}%). "
                      f"{highest_value_segment} have highest average value (₹{highest_avg_value:,.0f}). "
                      f"Champions: {champions_pct:.1f}% of customers.")
            
            self.insights.append(insight)
            
        except Exception as e:
            logger.error(f"Customer segmentation analysis failed: {e}")
            self.insights.append("Q18 Insight: Customer segmentation analysis failed due to data issues.")

    def q19_geographic_expansion(self):
        """Question 19: Geographic expansion opportunities analysis"""
        logger.info("EDA Question 19: Geographic expansion analysis")
        
        try:
            # City-wise performance analysis
            city_performance = self.df.groupby('customer_city').agg({
                'transaction_id': 'count',
                'final_amount_inr': 'sum',
                'customer_id': 'nunique',
                'customer_rating': 'mean'
            }).round(2)
            
            city_performance.columns = ['total_orders', 'total_revenue', 'unique_customers', 'avg_rating']
            city_performance['avg_order_value'] = city_performance['total_revenue'] / city_performance['total_orders']
            city_performance['orders_per_customer'] = city_performance['total_orders'] / city_performance['unique_customers']
            
            # Filter cities with sufficient data
            city_performance = city_performance[city_performance['total_orders'] > 100]
            
            # Growth analysis by city
            city_growth = self.df.groupby(['customer_city', 'order_year'])['final_amount_inr'].sum().unstack(fill_value=0)
            if len(city_growth.columns) > 1:
                city_growth['growth_rate'] = ((city_growth.iloc[:, -1] - city_growth.iloc[:, 0]) / city_growth.iloc[:, 0]) * 100
            else:
                city_growth['growth_rate'] = 0
            
            # Market penetration analysis
            total_indian_cities = 4000  # Approximate number of cities in India
            penetrated_cities = len(city_performance)
            penetration_rate = (penetrated_cities / total_indian_cities) * 100
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Top cities by revenue
            top_cities_revenue = city_performance.nlargest(15, 'total_revenue')['total_revenue'].sort_values()
            bars1 = ax1.barh(range(len(top_cities_revenue)), top_cities_revenue.values,
                           color='#1f77b4', alpha=0.7)
            ax1.set_yticks(range(len(top_cities_revenue)))
            ax1.set_yticklabels(top_cities_revenue.index)
            ax1.set_title('Top 15 Cities by Revenue', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Total Revenue (INR)')
            
            # Plot 2: Growth champions
            if 'growth_rate' in city_growth.columns:
                growth_champions = city_growth.nlargest(10, 'growth_rate')['growth_rate'].sort_values()
                if len(growth_champions) > 0:
                    bars2 = ax2.barh(range(len(growth_champions)), growth_champions.values,
                                   color='#2ca02c', alpha=0.7)
                    ax2.set_yticks(range(len(growth_champions)))
                    ax2.set_yticklabels(growth_champions.index)
                    ax2.set_title('Top 10 Growth Cities', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Growth Rate (%)')
                    ax2.axvline(x=0, color='red', linestyle='-', alpha=0.5)
            
            # Plot 3: Market penetration analysis
            penetrated = penetrated_cities
            untapped = total_indian_cities - penetrated_cities
            
            wedges3, texts3, autotexts3 = ax3.pie([penetrated, untapped], 
                                                 labels=['Penetrated Cities', 'Untapped Cities'],
                                                 autopct='%1.1f%%', colors=['#1f77b4', '#ff7f0e'], 
                                                 startangle=90)
            ax3.set_title(f'Market Penetration: {penetrated}/{total_indian_cities} Cities', 
                         fontsize=14, fontweight='bold')
            
            # Plot 4: Revenue density vs city size
            city_performance['revenue_density'] = city_performance['total_revenue'] / city_performance['unique_customers']
            top_density_cities = city_performance.nlargest(10, 'revenue_density')['revenue_density'].sort_values()
            
            bars4 = ax4.barh(range(len(top_density_cities)), top_density_cities.values,
                           color='#d62728', alpha=0.7)
            ax4.set_yticks(range(len(top_density_cities)))
            ax4.set_yticklabels(top_density_cities.index)
            ax4.set_title('Top 10 Cities by Revenue per Customer', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Revenue per Customer (INR)')
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'q19_geographic_expansion.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate insights
            top_city = top_cities_revenue.index[-1]
            top_city_revenue = top_cities_revenue.iloc[-1]
            fastest_growing_city = growth_champions.index[-1] if 'growth_champions' in locals() and len(growth_champions) > 0 else 'Unknown'
            fastest_growth = growth_champions.iloc[-1] if 'growth_champions' in locals() and len(growth_champions) > 0 else 0
            highest_density_city = top_density_cities.index[-1]
            highest_density = top_density_cities.iloc[-1]
            
            insight = (f"Q19 Insight: {top_city} generates highest revenue (₹{top_city_revenue:,.0f}). "
                      f"{fastest_growing_city} fastest growing ({fastest_growth:+.1f}%). "
                      f"{highest_density_city} has highest revenue density (₹{highest_density:,.0f}/customer). "
                      f"Market penetration: {penetration_rate:.2f}%.")
            
            self.insights.append(insight)
            
        except Exception as e:
            logger.error(f"Geographic expansion analysis failed: {e}")
            self.insights.append("Q19 Insight: Geographic expansion analysis failed due to data issues.")

    def q20_business_recommendations(self):
        """Question 20: Simple Business Health Dashboard"""
        logger.info("EDA Question 20: Creating Business Health Dashboard")
        
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Calculate basic metrics
            total_revenue = self.df['final_amount_inr'].sum()
            total_customers = self.df['customer_id'].nunique()
            avg_order_value = self.df['final_amount_inr'].mean()
            
            # Plot 1: Revenue Trend
            yearly_revenue = self.df.groupby('order_year')['final_amount_inr'].sum()
            axes[0,0].plot(yearly_revenue.index, yearly_revenue.values, marker='o', linewidth=2, color='blue')
            axes[0,0].set_title('Revenue Trend Over Years')
            axes[0,0].set_xlabel('Year')
            axes[0,0].set_ylabel('Revenue (INR)')
            axes[0,0].grid(True, alpha=0.3)
            
            # Plot 2: Top Subcategories
            subcategory_revenue = self.df.groupby('subcategory')['final_amount_inr'].sum().nlargest(6)
            axes[0,1].bar(subcategory_revenue.index, subcategory_revenue.values, color='green', alpha=0.7)
            axes[0,1].set_title('Top Subcategories by Revenue')
            axes[0,1].set_xlabel('Subcategory')
            axes[0,1].set_ylabel('Revenue (INR)')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Customer Growth
            yearly_customers = self.df.groupby('order_year')['customer_id'].nunique()
            axes[1,0].plot(yearly_customers.index, yearly_customers.values, marker='s', linewidth=2, color='red')
            axes[1,0].set_title('Customer Growth Over Years')
            axes[1,0].set_xlabel('Year')
            axes[1,0].set_ylabel('Number of Customers')
            axes[1,0].grid(True, alpha=0.3)
            
            # Plot 4: Payment Methods
            if 'payment_method' in self.df.columns:
                payment_counts = self.df['payment_method'].value_counts().head(5)
                axes[1,1].pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%')
                axes[1,1].set_title('Payment Method Distribution')
            
            # Add overall title
            plt.suptitle('Business Health Dashboard - Key Metrics Overview', fontsize=16, fontweight='bold')
            
            # Adjust layout and save
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'q20_business_health_dashboard.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Simple Business Health Dashboard created successfully")
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")

    def classify_city_tier(self, city_name):
        """Helper method to classify cities into tiers"""
        if pd.isna(city_name) or city_name is None:
            return 'Unknown'
        
        metro_cities = ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune']
        tier1_cities = ['ahmedabad', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'surat', 'indore', 
                    'coimbatore', 'kochi', 'vadodara', 'visakhapatnam', 'bhubaneswar']
        
        city_lower = str(city_name).lower().strip()
        
        if any(metro in city_lower for metro in metro_cities):
            return 'Metro'
        elif any(tier1 in city_lower for tier1 in tier1_cities):
            return 'Tier 1'
        else:
            return 'Tier 2/Rural'
            
    def run_all_analysis(self):
        """Execute all EDA analyses"""
        logger.info("Starting comprehensive EDA analysis")
        
        # Run all analysis methods in correct order
        analysis_methods = [
            self.q1_revenue_trend,                    # Q1: Revenue trend analysis
            self.q2_seasonal_patterns,                # Q2: Seasonal patterns
            self.q3_rfm_segmentation,                 # Q3: RFM customer segmentation
            self.q4_payment_method_evolution,         # Q4: Payment method evolution
            self.q5_subcategory_performance,          # Q5: Category performance
            self.q6_prime_membership_impact,          # Q6: Prime membership impact
            self.q7_geographic_analysis,              # Q7: Geographic analysis
            self.q8_festival_sales_impact,            # Q8: Festival sales impact
            self.q9_customer_age_group_analysis,      # Q9: Customer age group behavior - NEW
            self.q10_price_vs_demand_analysis,        # Q10: Price vs demand analysis - NEW
            self.q11_delivery_performance,            # Q11: Delivery performance
            self.q12_return_patterns,                 # Q12: Return patterns
            self.q13_brand_performance,               # Q13: Brand performance
            self.q14_customer_lifetime_value,         # Q14: Customer lifetime value
            self.q15_discount_effectiveness,          # Q15: Discount effectiveness
            self.q16_rating_impact_analysis,          # Q16: Rating impact analysis
            self.q17_prime_member_analysis,           # Q17: Prime member analysis
            self.q18_customer_segmentation,           # Q18: Customer segmentation
            self.q19_geographic_expansion,            # Q19: Geographic expansion
            self.q20_business_recommendations         # Q20: Business health dashboard
        ]
        
        for method in analysis_methods:
            try:
                method()
                logger.info(f"✅ Completed {method.__name__}")
            except Exception as e:
                logger.error(f"❌ Error in {method.__name__}: {e}")
                import traceback
                logger.error(f"🔍 Detailed error: {traceback.format_exc()}")
        
        logger.info("Completed all EDA analyses")
# Usage example
if __name__ == "__main__":
    # Update these paths according to your file locations
    data_path = "transaction_data.csv"  # Update with your actual file path
    product_data_path = "product_data.csv"  # Update with your actual file path
    output_dir = "eda_results"
    
    eda = EDAAnalysis(data_path, product_data_path, output_dir)
    eda.run_all_analysis()


