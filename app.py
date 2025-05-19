from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json
import os

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv('C:\\Users\\ASUS\\OneDrive\\Desktop\\AI ML PBL\\enhanced_productivity_finance_dataset_500_rows.csv')


# Prepare data for ML models
def prepare_data():
    # Productivity prediction
    X_prod = df[['Age', 'Daily Work Hours', 'Daily Leisure Hours', 
                'Daily Exercise Minutes', 'Daily Sleep Hours', 
                'Screen Time (hours)', 'Commute Time (hours)']]
    y_prod = df['Productivity Score']
    
    # Financial health prediction
    X_fin = df[['Monthly Income ($)', 'Debt-to-Income Ratio (%)', 
               'Investment Portfolio Size ($)', 'Credit Score (300-850)', 
               'Monthly Savings ($)']]
    y_fin = df['Budget Adherence (%)']
    
    # Scale features
    scaler_prod = StandardScaler().fit(X_prod)
    scaler_fin = StandardScaler().fit(X_fin)
    
    return {
        'prod': {'X': X_prod, 'y': y_prod, 'scaler': scaler_prod},
        'fin': {'X': X_fin, 'y': y_fin, 'scaler': scaler_fin}
    }

data = prepare_data()

prod_model = LinearRegression().fit(data['prod']['scaler'].transform(data['prod']['X']), data['prod']['y'])
fin_model = LinearRegression().fit(data['fin']['scaler'].transform(data['fin']['X']), data['fin']['y'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/productivity')
def productivity():
    return render_template('productivity.html')

@app.route('/finance')
def finance():
    return render_template('finance.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html')

@app.route('/api/predict_productivity', methods=['POST'])
def predict_productivity():
    try:
        user_data = request.json
        input_data = np.array([
            user_data['age'],
            user_data['work_hours'],
            user_data['leisure_hours'],
            user_data['exercise_minutes'],
            user_data['sleep_hours'],
            user_data['screen_time'],
            user_data['commute_time']
        ]).reshape(1, -1)
        
        scaled_data = data['prod']['scaler'].transform(input_data)
        prediction = prod_model.predict(scaled_data)[0]
        
        # Generate recommendations
        recommendations = []
        if user_data['sleep_hours'] < 7:
            recommendations.append("Increase your sleep to at least 7 hours for better productivity")
        if user_data['exercise_minutes'] < 30:
            recommendations.append("Try to get at least 30 minutes of exercise daily")
        if user_data['screen_time'] > 6:
            recommendations.append("Consider reducing screen time to reduce eye strain and improve focus")
            
        return jsonify({
            'predicted_score': round(prediction, 1),
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict_finance', methods=['POST'])
def predict_finance():
    try:
        user_data = request.json
        input_data = np.array([
            user_data['income'],
            user_data['debt_ratio'],
            user_data['investments'],
            user_data['credit_score'],
            user_data['monthly_savings']
        ]).reshape(1, -1)
        
        scaled_data = data['fin']['scaler'].transform(input_data)
        prediction = fin_model.predict(scaled_data)[0]
        
        # Generate recommendations
        recommendations = []
        if user_data['debt_ratio'] > 30:
            recommendations.append("Your debt-to-income ratio is high. Consider paying down debts.")
        if user_data['monthly_savings']/user_data['income'] < 0.2:
            recommendations.append("Try to save at least 20% of your income each month")
        if user_data['investments'] < user_data['income'] * 5:
            recommendations.append("Consider increasing your investments for long-term financial health")
            
        return jsonify({
            'predicted_adherence': round(prediction, 1),
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/get_insights')
def get_insights():
    # Calculate some statistics from the dataset
    avg_productivity = round(df['Productivity Score'].mean(), 1)
    avg_savings = round(df['Monthly Savings ($)'].mean(), 2)
    common_screen_time = round(df['Screen Time (hours)'].mean(), 1)
    
    return jsonify({
        'avg_productivity': avg_productivity,
        'avg_savings': avg_savings,
        'common_screen_time': common_screen_time,
        'ai_schedule_users': int(df['AI Schedule Suggestions (Y/N)'].value_counts().get('Y', 0)),
        'spending_alert_users': int(df['Real-Time Spending Alerts (Y/N)'].value_counts().get('Y', 0))
    })

if __name__ == '__main__':
    app.run(debug=True)