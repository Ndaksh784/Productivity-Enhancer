import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
  
    df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\AI ML PBL\enhanced_productivity_finance_dataset_500_rows.csv')

    
    X_prod = df[['Age', 'Daily Work Hours', 'Daily Leisure Hours', 
                 'Daily Exercise Minutes', 'Daily Sleep Hours', 
                 'Screen Time (hours)', 'Commute Time (hours)']]
    y_prod = df['Productivity Score']

   
    X_fin = df[['Monthly Income ($)', 'Debt-to-Income Ratio (%)', 
                'Investment Portfolio Size ($)', 'Credit Score (300-850)', 
                'Monthly Savings ($)']]
    y_fin = df['Budget Adherence (%)']

   
    scaler_prod = StandardScaler().fit(X_prod)
    scaler_fin = StandardScaler().fit(X_fin)

    return {
        'prod': {'X': X_prod, 'y': y_prod, 'scaler': scaler_prod},
        'fin': {'X': X_fin, 'y': y_fin, 'scaler': scaler_fin},
        'df': df 
    }

def train_models(data):
    
    prod_model = LinearRegression()
    prod_model.fit(data['prod']['scaler'].transform(data['prod']['X']), data['prod']['y'])

   
    fin_model = LinearRegression()
    fin_model.fit(data['fin']['scaler'].transform(data['fin']['X']), data['fin']['y'])

    return prod_model, fin_model

def get_dataset_insights(df):
    return {
        'avg_productivity': round(df['Productivity Score'].mean(), 1),
        'avg_savings': round(df['Monthly Savings ($)'].mean(), 2),
        'common_screen_time': round(df['Screen Time (hours)'].mean(), 1),
        'ai_schedule_users': int(df['AI Schedule Suggestions (Y/N)'].value_counts().get('Y', 0)),
        'spending_alert_users': int(df['Real-Time Spending Alerts (Y/N)'].value_counts().get('Y', 0))
    }
