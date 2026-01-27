"""
PROJECT: E-Commerce Customer & Revenue Analysis

BUSINESS PROBLEM:
An e-commerce company wants to understand customer behavior
to increase revenue and retain valuable customers.

BUSINESS QUESTIONS:
1. Who are the high-value customers?
2. What are the purchasing patterns over time?
3. Which products drive the most revenue?
4. How can the company improve customer retention?
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
df=pd.read_csv("data/raw_data.csv")
print(df.head())
print(df.info())

# Columns
# InvoiceNo -> Order identifier
# Stock code -> Product Identifier
# Description -> Product name (Categorical)
# Quantity -> Number of items bought (Numerical)
# Invoice date -> date_time of purchases (datetime)
# Unit price -> price per item (Numerical)
# Customer_id -> Customer identifier
# Country -> Customer country (Categorical)


# =========================
# DATA CLEANING
# =========================

# Remove cancelled orders (InvoiceNo starts with 'C')
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Missing values
missing_values=df.isnull().sum()

# Drop missing CustomerID
df = df.dropna(subset=["CustomerID"])

# Fix data types
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["CustomerID"] = df["CustomerID"].astype(int)

# Remove negative quantities
df = df[df["Quantity"] > 0]

# Clean description text
df["Description"] = df["Description"].str.strip()

# Create revenue column
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

print("Cleaned data shape:", df.shape)



# =========================
# MONTHLY REVENUE TREND
# =========================

df["Month"] = df["InvoiceDate"].dt.to_period("M")

monthly_revenue = df.groupby("Month")["Revenue"].sum()

plt.figure(figsize=(10, 5))
monthly_revenue.plot()
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.tight_layout()

plt.savefig("visuals/monthly_revenue.png")
plt.show()



# =========================
# TOP PRODUCTS BY REVENUE
# =========================

top_products = (
    df.groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(top_products)



# =========================
# CUSTOMER PURCHASE BEHAVIOR
# =========================

orders_per_customer = df.groupby("CustomerID")["InvoiceNo"].nunique()

repeat_customers = (orders_per_customer > 1).sum()
one_time_customers = (orders_per_customer == 1).sum()

print("Repeat customers:", repeat_customers)
print("One-time customers:", one_time_customers)



# =========================
# RFM ANALYSIS
# =========================

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "Revenue": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

# Simple VIP segmentation
rfm["Segment"] = "Regular"
rfm.loc[
    (rfm["Recency"] <= 30) &
    (rfm["Frequency"] >= 5) &
    (rfm["Monetary"] >= rfm["Monetary"].quantile(0.75)),
    "Segment"
] = "VIP"

print(rfm["Segment"].value_counts())



"""
KEY INSIGHTS:
1. Revenue shows seasonal patterns with peaks toward year-end.
2. A small group of VIP customers generates a large share of revenue.
3. Many customers purchase only once, indicating retention opportunity.
4. Certain products consistently drive revenue.

RECOMMENDATIONS:
1. Create loyalty programs for VIP customers.
2. Run re-engagement campaigns for inactive customers.
3. Focus inventory on top-performing products.
"""
