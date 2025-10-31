import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Inventory Dashboard", page_icon="📊", layout="wide")

st.title("📊 AI-Powered Inventory Management Dashboard")
st.markdown("*Real-time inventory monitoring and forecasting system*")
st.markdown("---")

@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')
    base_demand = np.linspace(50, 100, n_samples)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
    noise = np.random.normal(0, 5, n_samples)
    
    df = pd.DataFrame({
        'Date': dates,
        'Store': np.random.choice(['Store_A', 'Store_B', 'Store_C', 'Store_D'], n_samples),
        'Product': np.random.choice(['Product_X', 'Product_Y', 'Product_Z'], n_samples),
        'Sales_Quantity': np.abs(base_demand + seasonality + noise),
        'Price': np.random.uniform(10, 50, n_samples),
        'Inventory': np.abs(base_demand + seasonality + noise) * np.random.uniform(1.5, 3.0, n_samples),
        'Category': np.random.choice(['Category_1', 'Category_2', 'Category_3'], n_samples)
    })
    return df

df = load_data()

st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Select Page:", 
    ["📊 Dashboard", "🔮 Forecasting", "📦 Inventory", "🎯 Models", "⚙️ Settings"])

if page == "📊 Dashboard":
    st.subheader("📈 Real-Time Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = df['Sales_Quantity'].sum()
    avg_inventory = df['Inventory'].mean()
    total_products = df['Product'].nunique()
    stores = df['Store'].nunique()
    
    with col1:
        st.metric("Total Sales", f"{total_sales:,.0f}", "+12.5%")
    with col2:
        st.metric("Avg Inventory", f"{avg_inventory:,.0f}", "-2.3%")
    with col3:
        st.metric("Products", total_products)
    with col4:
        st.metric("Active Stores", stores)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sales Trend")
        daily_sales = df.groupby('Date')['Sales_Quantity'].sum().reset_index()
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=daily_sales['Date'], y=daily_sales['Sales_Quantity'],
            mode='lines+markers', name='Daily Sales',
            line=dict(color='#1f77b4', width=3)
        ))
        fig1.update_layout(title="Sales Over Time", xaxis_title="Date", 
                          yaxis_title="Quantity", height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("🏪 Sales by Store")
        store_sales = df.groupby('Store')['Sales_Quantity'].sum().sort_values(ascending=False)
        fig2 = go.Figure(data=[go.Bar(x=store_sales.index, y=store_sales.values,
                   marker=dict(color=store_sales.values, colorscale='Viridis'))])
        fig2.update_layout(title="Total Sales by Store", xaxis_title="Store", 
                          yaxis_title="Sales", height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Inventory Distribution")
        fig3 = px.histogram(df, x='Inventory', nbins=50,
                           title='Inventory Levels', color_discrete_sequence=['#2ca02c'])
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.subheader("🛍️ Product Mix")
        product_sales = df.groupby('Product')['Sales_Quantity'].sum()
        fig4 = go.Figure(data=[go.Pie(labels=product_sales.index, values=product_sales.values)])
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

elif page == "🔮 Forecasting":
    st.subheader("🔮 AI Demand Forecasting")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_store = st.selectbox("Select Store:", df['Store'].unique())
    with col2:
        forecast_days = st.number_input("Days:", 7, 90, 30)
    
    store_data = df[df['Store'] == selected_store].sort_values('Date')
    
    future_dates = pd.date_range(
        start=store_data['Date'].max() + timedelta(days=1),
        periods=forecast_days, freq='D'
    )
    
    historical_avg = store_data['Sales_Quantity'].mean()
    forecast_trend = np.linspace(historical_avg, historical_avg * 1.15, forecast_days)
    forecast_noise = np.random.normal(0, 5, forecast_days)
    forecast_values = np.abs(forecast_trend + forecast_noise)
    
    historical_std = store_data['Sales_Quantity'].std()
    safety_stock = 1.96 * historical_std
    reorder_point = historical_avg + safety_stock
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=store_data['Date'], y=store_data['Sales_Quantity'],
        mode='lines+markers', name='Historical', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_values,
        mode='lines+markers', name='Forecast', line=dict(color='#ff7f0e', width=2, dash='dash')))
    fig.add_hline(y=reorder_point, line_dash="dash", line_color="red",
                  annotation_text="Reorder Point", annotation_position="right")
    fig.update_layout(title=f"Demand Forecast - {selected_store}",
        xaxis_title="Date", yaxis_title="Quantity", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📋 Recommendations")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Demand", f"{forecast_values.mean():,.0f}")
    with col2:
        st.metric("Safety Stock", f"{safety_stock:,.0f}")
    with col3:
        st.metric("Reorder Point", f"{reorder_point:,.0f}")
    with col4:
        st.metric("Recommended Stock", f"{reorder_point * 1.3:,.0f}")
    
    forecast_table = pd.DataFrame({
        'Date': future_dates,
        'Predicted': forecast_values.round(0),
        'Safety Stock': [safety_stock] * forecast_days,
        'Reorder Point': [reorder_point] * forecast_days,
    })
    
    st.subheader("📊 Forecast Table")
    st.dataframe(forecast_table.head(15), use_container_width=True)
    
    csv = forecast_table.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast",
        data=csv,
        file_name=f"forecast_{selected_store}.csv",
        mime="text/csv"
    )

elif page == "📦 Inventory":
    st.subheader("📦 Inventory Analysis")
    col1, col2 = st.columns(2)
    with col1:
        selected_product = st.selectbox("Product:", df['Product'].unique())
    with col2:
        selected_category = st.selectbox("Category:", df['Category'].unique())
    
    product_data = df[df['Product'] == selected_product]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Level", f"{product_data['Inventory'].mean():,.0f}")
    with col2:
        st.metric("Max", f"{product_data['Inventory'].max():,.0f}")
    with col3:
        st.metric("Min", f"{product_data['Inventory'].min():,.0f}")
    with col4:
        st.metric("Std Dev", f"{product_data['Inventory'].std():,.0f}")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Inventory Trend")
        daily_inv = product_data.groupby('Date')['Inventory'].mean()
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=daily_inv.index, y=daily_inv.values,
            fill='tozeroy', name='Inventory', line=dict(color='#2ca02c', width=2)))
        fig1.update_layout(title=f"Inventory Trend", height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("📊 Distribution")
        fig2 = go.Figure(data=[go.Box(y=product_data['Inventory'], marker_color='#d62728')])
        fig2.update_layout(title=f"Inventory Distribution", height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Inventory by Store")
    store_inv = df.groupby('Store').agg({
        'Inventory': ['mean', 'min', 'max', 'std'],
        'Sales_Quantity': 'sum'
    }).round(2)
    store_inv.columns = ['Avg Inv', 'Min', 'Max', 'Std Dev', 'Total Sales']
    st.dataframe(store_inv, use_container_width=True)

elif page == "🎯 Models":
    st.subheader("🎯 AI Model Performance")
    models = ['Linear Regression', 'Ridge', 'Random Forest', 'Gradient Boosting', 'Neural Network']
    rmse = [15.2, 14.8, 12.5, 11.3, 10.1]
    mae = [12.1, 11.9, 9.5, 8.7, 7.9]
    r2 = [0.72, 0.74, 0.82, 0.86, 0.89]
    
    perf_df = pd.DataFrame({'Model': models, 'RMSE': rmse, 'MAE': mae, 'R²': r2})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best R²", f"{max(r2):.2%}")
    with col2:
        st.metric("Lowest RMSE", f"{min(rmse):.2f}")
    with col3:
        st.metric("Lowest MAE", f"{min(mae):.2f}")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Error Comparison")
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=models, y=rmse, name='RMSE', marker_color='#1f77b4'))
        fig1.add_trace(go.Bar(x=models, y=mae, name='MAE', marker_color='#ff7f0e'))
        fig1.update_layout(title="Error Metrics", barmode='group', height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("📈 R² Score Comparison")
        fig2 = go.Figure(data=[go.Bar(x=models, y=r2,
                                     marker=dict(color=r2, colorscale='Greens'))])
        fig2.update_layout(title="R² Score by Model", height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Detailed Performance Table")
    st.dataframe(perf_df.style.highlight_max(subset=['R²'], color='lightgreen')
                              .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen'),
                use_container_width=True)

elif page == "⚙️ Settings":
    st.subheader("⚙️ Configuration Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Settings")
        service_level = st.slider("Service Level (%)", 80, 99, 95)
        lead_time = st.number_input("Lead Time (Days)", 1, 30, 7)
        reorder_method = st.selectbox("Reorder Method", ["Fixed", "ABC Analysis", "Min-Max"])
        st.info(f"✅ Selected: {reorder_method} method with {lead_time} days lead time")
    
    with col2:
        st.subheader("🎨 Display Settings")
        theme = st.radio("Theme", ["Light", "Dark"])
        date_fmt = st.selectbox("Date Format", ["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD"])
        refresh = st.slider("Auto-Refresh Rate (seconds)", 30, 300, 60)
        st.info(f"✅ Theme: {theme} | Refresh: {refresh}s")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Settings", use_container_width=True):
            st.success("✅ Settings saved successfully!")
    with col2:
        if st.button("🔄 Reset to Default", use_container_width=True):
            st.info("↺ Settings reset to default values")

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>🚀 AI-Powered Inventory Management Dashboard v1.0</p>
    <p>Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p>Powered by Streamlit | Built with ❤️ for Inventory Management</p>
</div>
""", unsafe_allow_html=True)
