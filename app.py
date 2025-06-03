import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import plotly.express as px
import io

# Set page configuration
st.set_page_config(
    page_title="Carnival Wars - Price Predictor",
    page_icon="💰",
    layout="wide"
)

st.markdown("""
    <style>
    /* Hide Streamlit default menu and footer */
    #MainMenu, footer {
        visibility: hidden;
    }
    /* Hide any header that isn’t the hero-banner */
    header:not(.hero-banner) {
        visibility: hidden;
    }

    /* ===== WebKit Browsers (Chrome, Safari, Edge, Opera) ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }

    /* ===== Firefox ===== */
    html {
        scrollbar-width: thin;
        scrollbar-color: rgba(255, 255, 255, 0.15) transparent;
    }

    /* Main container constraints */
    .main .block-container {
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
        padding: 2rem 4rem;
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }

    /* Heading styles */
    h1 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    h2 {
        font-size: 1.25rem;
        margin-bottom: 0.75rem;
    }
    h3 {
        font-size: 1rem;
        font-weight: normal;
        margin-bottom: 0.5rem;
    }

    /* Hero banner styling */
    .hero-banner {
        background: linear-gradient(to bottom, #1a1a2e 0%, #16213e 100%);
        padding: 2rem 1rem;
        text-align: center;
        margin: 1rem auto;
        width: 95%;
        max-width: 600px;
        color: white;
        border-radius: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        border: 1px solid #4a90e2;
    }
    .hero-banner h1 {
        font-size: clamp(1.5rem, 5vw, 1.75rem);
        font-weight: bold;
        margin: 0;
    }
    .hero-banner p {
        font-size: clamp(0.75rem, 2.5vw, 0.875rem);
        font-weight: normal;
        margin: 0.5rem 0;
    }

    /* Predicted price block styling */
    .price-block {
        background: linear-gradient(to bottom, #2e7d32 0%, #1b5e20 80%);
        padding: 2rem;
        text-align: center;
        margin: 1.5rem auto;
        width: 90%;
        max-width: 300px;
        color: white;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        border: 1px solid #4caf50;
    }
    .price-block h2 {
        font-size: 1rem;
        font-weight: bold;
        margin: 0 0 0.2rem 0;
        color: #f5f5f5;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    .price-block p {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
    }

    /* Footer styling */
    .footer {
        order: 3;
        margin-top: 1rem;
        padding: 1rem 0;
        text-align: center;
        border-top: 1px solid #434343;
    }
    .footer p {
        margin: 0;
        font-size: 14px;
        color: #999;
    }
    .footer a {
        color: #1e90ff;
        text-decoration: none;
    }

    /* Tab styling */
    .stTabs [role="tab"] {
        background: #1a1a2e;
        color: white;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: #16213e;
        color: #4a90e2;
    }

    /* Plotly chart container styling */
    .stPlotlyChart > div {
        border-radius: 15px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
        padding: 15px !important;
        border: 1px solid #4a90e2 !important;
        background-color: #16213e !important;
    }
    </style>
""", unsafe_allow_html=True)



def load_model():
    with st.spinner("Loading model..."):
        try:
            model = joblib.load("resources/stacked_model.pkl")
            feature_names = [
                'Stall_no', 'Market_Category', 'Loyalty_customer', 'Grade', 'Demand', 'Discount_avail',
                'charges_1', 'charges_2 (%)', 'Minimum_price', 'Maximum_price', 'day', 'month', 'year',
                'dayofweek', 'weekofyear', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'charges_1_missing', 'charges_2 (%)_missing', 'Minimum_price_missing', 'Maximum_price_missing',
                'price_range', 'price_ratio', 'log_charges_1',
                'Product_Category_Cosmetics', 'Product_Category_Educational', 'Product_Category_Fashion',
                'Product_Category_Home_decor', 'Product_Category_Hospitality', 'Product_Category_Organic',
                'Product_Category_Pet_care', 'Product_Category_Repair', 'Product_Category_Technology'
            ]
            return {
                'model': model,
                'feature_names': feature_names
            }
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None


def preprocess_input(input_data, product_category):
    df = input_data.copy()
    df['instock_date'] = pd.to_datetime(df['instock_date'])
    df['day'] = df['instock_date'].dt.day
    df['month'] = df['instock_date'].dt.month
    df['year'] = df['instock_date'].dt.year
    df['dayofweek'] = df['instock_date'].dt.dayofweek
    df['weekofyear'] = df['instock_date'].dt.isocalendar().week.astype(int)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    for col in ['charges_1', 'charges_2 (%)', 'Minimum_price', 'Maximum_price']:
        df[f'{col}_missing'] = 0
    df['price_range'] = df['Maximum_price'] - df['Minimum_price']
    df['price_ratio'] = df['charges_1'] / (df['Maximum_price'] + 1)
    df['log_charges_1'] = np.log1p(df['charges_1'])
    market_category_map = {'Small': 0, 'Medium': 1, 'Large': 2}
    df['Market_Category'] = df['Market_Category'].map(market_category_map)
    grade_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Grade'] = df['Grade'].map(grade_map)
    demand_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Demand'] = df['Demand'].map(demand_map)
    df['Product_Category_Cosmetics'] = 1 if product_category == 'Cosmetics' else 0
    df['Product_Category_Educational'] = 1 if product_category == 'Educational' else 0
    df['Product_Category_Fashion'] = 1 if product_category == 'Fashion' else 0
    df['Product_Category_Home_decor'] = 1 if product_category == 'Home Decor' else 0
    df['Product_Category_Hospitality'] = 1 if product_category == 'Hospitality' else 0
    df['Product_Category_Organic'] = 1 if product_category == 'Organic' else 0
    df['Product_Category_Pet_care'] = 1 if product_category == 'Pet Care' else 0
    df['Product_Category_Repair'] = 1 if product_category == 'Repair' else 0
    df['Product_Category_Technology'] = 1 if product_category == 'Technology' else 0
    expected_columns = [
        'Stall_no', 'Market_Category', 'Loyalty_customer', 'Grade', 'Demand', 'Discount_avail',
        'charges_1', 'charges_2 (%)', 'Minimum_price', 'Maximum_price', 'day', 'month', 'year',
        'dayofweek', 'weekofyear', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'charges_1_missing', 'charges_2 (%)_missing', 'Minimum_price_missing', 'Maximum_price_missing',
        'price_range', 'price_ratio', 'log_charges_1',
        'Product_Category_Cosmetics', 'Product_Category_Educational', 'Product_Category_Fashion',
        'Product_Category_Home_decor', 'Product_Category_Hospitality', 'Product_Category_Organic',
        'Product_Category_Pet_care', 'Product_Category_Repair', 'Product_Category_Technology'
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                df = pd.get_dummies(df, columns=[col], drop_first=True)
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0.0
    df = df[expected_columns]
    return df.astype(float)


def main():
    model_data = load_model()
    if model_data is None:
        st.error("Failed to load the model. Please ensure the model file exists.")
        st.stop()

    # Hero banner
    st.markdown("""
        <header class="hero-banner" role="banner" aria-label="Application Header">
            <h1>🎪 Carnival Wars - Product Price Predictor</h1>
            <p>Predict the optimal selling price for your products</p>
        </header>
    """, unsafe_allow_html=True)
    st.markdown("<!-- Hero banner should appear above -->", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Predict Price", "Batch Prediction"])

    with tab1:
        st.header("Single Product Prediction")
        with st.form("prediction_form"):
            st.subheader("Product Details")
            col1, col2 = st.columns(2)
            with col1:
                product_category = st.selectbox(
                    "Product Category",
                    ["Cosmetics", "Educational", "Fashion", "Home Decor",
                     "Hospitality", "Organic", "Pet Care", "Repair", "Technology"],
                    help="Select the category of the product."
                )
                stall_no = st.number_input(
                    "Stall Number",
                    min_value=1,
                    max_value=100,
                    value=10,
                    help="The stall number in the market (1-100)."
                )
            with col2:
                market_category = st.selectbox(
                    "Market Category",
                    ["Small", "Medium", "Large"],
                    index=0,
                    help="The size of the market."
                )
                grade = st.selectbox(
                    "Grade",
                    ["Low", "Medium", "High"],
                    index=1,
                    help="The quality grade of the product."
                )

            st.subheader("Pricing Details")
            col3, col4 = st.columns(2)
            with col3:
                min_price = st.number_input(
                    "Minimum Price",
                    min_value=0.0,
                    value=1000.0,
                    step=100.0,
                    help="The minimum price for the product ($)."
                )
                max_price = st.number_input(
                    "Maximum Price",
                    min_value=0.0,
                    value=2000.0,
                    step=100.0,
                    help="The maximum price for the product ($)."
                )
                if max_price <= min_price:
                    st.warning("Maximum Price must be greater than Minimum Price.")
            with col4:
                charges_1 = st.number_input(
                    "Base Charges",
                    min_value=0.0,
                    value=100.0,
                    step=10.0,
                    help="The fixed cost associated with the product ($)."
                )
                charges_2 = st.number_input(
                    "Percentage Fee (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=0.5,
                    help="The percentage-based fee applied to the product."
                ) / 100

            with st.expander("Additional Details"):
                col5, col6 = st.columns(2)
                with col5:
                    loyalty_customer = st.checkbox("Loyalty Customer", help="Check if the customer is a loyalty member.")
                    discount_avail = st.checkbox("Discount Available", help="Check if a discount is available.")
                with col6:
                    demand = st.selectbox(
                        "Demand Level",
                        ["Low", "Medium", "High"],
                        index=1,
                        help="The demand level for the product."
                    )
                    instock_date = st.date_input(
                        "In-stock Date",
                        value=datetime.now(),
                        max_value=datetime.now(),
                        help="The date the product was stocked."
                    )

            submit_button = st.form_submit_button("Predict Price", args={"aria-label": "Predict Price Button"})

        if submit_button:
            if max_price <= min_price:
                st.error("Please ensure Maximum Price is greater than Minimum Price.")
            else:
                input_data = {
                    'Stall_no': float(stall_no),
                    'Market_Category': market_category,
                    'Demand': demand,
                    'instock_date': instock_date,
                    'charges_1': float(charges_1),
                    'charges_2 (%)': float(charges_2 * 100),
                    'Minimum_price': float(min_price),
                    'Maximum_price': float(max_price),
                    'Discount_avail': 1.0 if discount_avail else 0.0,
                    'Loyalty_customer': 1.0 if loyalty_customer else 0.0,
                    'Grade': grade,
                    'year': float(instock_date.year),
                    'month': float(instock_date.month),
                    'day': float(instock_date.day),
                    'dayofweek': float(instock_date.weekday()),
                    'weekofyear': float(instock_date.isocalendar().week)
                }
                df = pd.DataFrame([input_data])
                try:
                    processed_df = preprocess_input(df, product_category)
                    expected_features = model_data['feature_names']
                    for feature in expected_features:
                        if feature not in processed_df.columns:
                            processed_df[feature] = 0.0
                    processed_df = processed_df[expected_features]
                    lgb_pred = model_data['model']['lgb'].predict(processed_df)[0]
                    xgb_pred = model_data['model']['xgb'].predict(processed_df)[0]
                    cat_pred = model_data['model']['cat'].predict(processed_df)[0]
                    meta_input = np.array([[lgb_pred, xgb_pred, cat_pred]])
                    final_pred = model_data['model']['meta_model'].predict(meta_input)[0]
                    # Display predicted price as a styled block
                    st.markdown(f"""
                        <div class="price-block">
                            <h2>Predicted Selling Price</h2>
                            <p>${final_pred:,.2f}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<h2 style='text-align: center; margin: 1.5rem 0;'>Individual Model Predictions</h2>", unsafe_allow_html=True)
                    preds = {
                        'Model': ['LightGBM', 'XGBoost', 'CatBoost', 'Final Prediction'],
                        'Predicted Price': [lgb_pred, xgb_pred, cat_pred, final_pred]
                    }
                    fig = px.bar(
                        preds,
                        x='Model',
                        y='Predicted Price',
                        title='Model Predictions Comparison',
                        labels={'Predicted Price': 'Predicted Price ($)'},
                        color='Model',
                        text_auto='.2f',
                        height=400
                    )
                    fig.update_traces(
                        textposition='outside',
                        textfont=dict(size=12, color='white')
                    )
                    fig.update_layout(
                        title=dict(
                            text='Model Predictions Comparison',
                            x=0.34,
                            y=0.94,
                            font=dict(size=16, color='white')
                        ),
                        xaxis_title='',
                        yaxis_title='Predicted Price ($)',
                        yaxis=dict(
                            range=[0, max(preds['Predicted Price']) * 1.3],
                            fixedrange=True,
                            titlefont=dict(size=12, color='white'),
                            tickfont=dict(size=12, color='white')
                        ),
                        xaxis=dict(
                            fixedrange=True,
                            tickfont=dict(size=12, color='white')
                        ),
                        margin=dict(t=80, b=80, l=60, r=60),
                        plot_bgcolor='#1a1a2e',
                        paper_bgcolor='#16213e',
                        font=dict(color='white'),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    with tab2:
        st.header("Batch Prediction")
        st.write("Upload a CSV file containing product details to get predictions for multiple products at once.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload a CSV file with the required columns.")
        if uploaded_file is not None:
            if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
                st.error("File size exceeds 10MB. Please upload a smaller file.")
            else:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.subheader("Uploaded Data")
                    st.dataframe(batch_df.head())
                    required_columns = ['Product_Category', 'Stall_no', 'Market_Category', 'Grade', 'Demand',
                                       'Discount_avail', 'charges_1', 'charges_2 (%)', 'Minimum_price', 'Maximum_price',
                                       'instock_date']
                    missing_columns = [col for col in required_columns if col not in batch_df.columns]
                    if missing_columns:
                        st.error(f"Error: The uploaded file is missing the following required columns: {', '.join(missing_columns)}")
                    else:
                        predictions = []
                        progress_bar = st.progress(0)
                        total_rows = len(batch_df)
                        for idx, (_, row) in enumerate(batch_df.iterrows()):
                            try:
                                instock_date = row['instock_date']
                                if isinstance(instock_date, str):
                                    try:
                                        instock_date = pd.to_datetime(instock_date).date()
                                    except:
                                        instock_date = pd.to_datetime('today').date()
                                input_data = {
                                    'Stall_no': float(row['Stall_no']),
                                    'Market_Category': row['Market_Category'],
                                    'Loyalty_customer': 1.0 if str(row.get('Loyalty_customer', 'No')).strip().lower() in ['yes', '1', 'true'] else 0.0,
                                    'Grade': row['Grade'],
                                    'Demand': row['Demand'],
                                    'Discount_avail': 1.0 if str(row.get('Discount_avail', 'No')).strip().lower() in ['yes', '1', 'true'] else 0.0,
                                    'charges_1': float(row['charges_1']),
                                    'charges_2 (%)': float(row['charges_2 (%)']),
                                    'Minimum_price': float(row['Minimum_price']),
                                    'Maximum_price': float(row['Maximum_price']),
                                    'instock_date': instock_date
                                }
                                processed_df = preprocess_input(pd.DataFrame([input_data]), row['Product_Category'])
                                lgb_pred = model_data['model']['lgb'].predict(processed_df)[0]
                                xgb_pred = model_data['model']['xgb'].predict(processed_df)[0]
                                cat_pred = model_data['model']['cat'].predict(processed_df)[0]
                                meta_input = np.array([[lgb_pred, xgb_pred, cat_pred]])
                                final_pred = model_data['model']['meta_model'].predict(meta_input)[0]
                                predictions.append(final_pred)
                            except Exception as e:
                                st.error(f"Error processing row {idx + 2}: {str(e)}")
                                predictions.append(None)
                            progress_bar.progress((idx + 1) / total_rows)
                        batch_df['Predicted_Price'] = predictions
                        st.subheader("Batch Prediction Results")
                        st.dataframe(batch_df)
                        csv = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Error reading the CSV file: {str(e)}")

    # Footer
    st.markdown("""
        <footer class="footer">
            <p>Built with ❤️ by <a href="https://github.com/omkarbhad" target="_blank">Omkar Bhad</a> using Streamlit</p>
        </footer>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
