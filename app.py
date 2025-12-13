import pandas as pd
from prophet import Prophet
import plotly.express as px
import streamlit as st

# 1. Load dataset
df = pd.read_csv("Prophet_AQI.csv")

# 2. Ensure ds column is datetime
df['ds'] = pd.to_datetime(df['ds'])

# 3. Train separate models for regressors
df_tavg = df[['ds', 'tavg']].rename(columns={'tavg': 'y'})
model_tavg = Prophet()
model_tavg.fit(df_tavg)

df_wspd = df[['ds', 'wspd']].rename(columns={'wspd': 'y'})
model_wspd = Prophet()
model_wspd.fit(df_wspd)

df_prcp = df[['ds', 'prcp']].rename(columns={'prcp': 'y'})
model_prcp = Prophet()
model_prcp.fit(df_prcp)

# 4. Streamlit UI
st.title("AQI Prediction - Prayagraj")
target_date = st.date_input("Enter prediction date:")
target_dt = pd.to_datetime(target_date)

# 5. Days to forecast
last_date = df['ds'].max()
days_needed = (target_dt - last_date).days

if days_needed < 0:
    st.error("Please select a future date!")
else:
    # 6. Forecast regressors
    future_tavg = model_tavg.make_future_dataframe(periods=days_needed)
    forecast_tavg = model_tavg.predict(future_tavg)

    future_wspd = model_wspd.make_future_dataframe(periods=days_needed)
    forecast_wspd = model_wspd.predict(future_wspd)

    future_prcp = model_prcp.make_future_dataframe(periods=days_needed)
    forecast_prcp = model_prcp.predict(future_prcp)

    # 7. AQI model with regressors
    df_aqi = df[['ds', 'AQI', 'tavg', 'wspd', 'prcp']].rename(columns={'AQI': 'y'})
    model_aqi = Prophet()
    model_aqi.add_regressor('tavg')
    model_aqi.add_regressor('wspd')
    model_aqi.add_regressor('prcp')
    model_aqi.fit(df_aqi)

    # 8. Future AQI dataframe
    future_aqi = model_aqi.make_future_dataframe(periods=days_needed)
    future_aqi['tavg'] = forecast_tavg['yhat'].tail(len(future_aqi)).values
    future_aqi['wspd'] = forecast_wspd['yhat'].tail(len(future_aqi)).values
    future_aqi['prcp'] = forecast_prcp['yhat'].tail(len(future_aqi)).values

    # 9. Predict AQI
    forecast_aqi = model_aqi.predict(future_aqi)
    pred = forecast_aqi[forecast_aqi['ds'] == target_dt]

    if not pred.empty:
        pred_value = pred['yhat'].values[0]
        lower = pred['yhat_lower'].values[0]
        upper = pred['yhat_upper'].values[0]

        st.success(f"Predicted AQI: {pred_value:.2f}")
        st.info(f"Range: {lower:.2f} - {upper:.2f}")

    # 10. Plot chart
    # Plot both historical and forecast AQI
fig = px.line(forecast_aqi, x='ds', y='yhat', title='AQI Forecast')

# Overlay actual historical AQI values
fig.add_scatter(x=df['ds'], y=df['AQI'], mode='lines', name='Historical')

# Optional: add confidence interval bounds
fig.add_scatter(x=forecast_aqi['ds'], y=forecast_aqi['yhat_lower'],
                mode='lines', line=dict(dash='dot', color='gray'),
                name='Lower bound')
fig.add_scatter(x=forecast_aqi['ds'], y=forecast_aqi['yhat_upper'],
                mode='lines', line=dict(dash='dot', color='gray'),
                name='Upper bound')

st.plotly_chart(fig, use_container_width=True)
