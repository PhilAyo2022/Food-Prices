import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# load data
@st.cache_data
def load_data():
    

    food = pd.read_csv('africa_food_prices.csv')
    food.drop(columns=["Unnamed: 0","country_id","state_id","market_id",
                   "produce_id","currency_id","pt_id",
                   "mp_commoditysource","quantity"],
        inplace = True)
    food.rename({'um_unit_id':'goods_purchased'}, axis=1, inplace=True)
    food.rename({'state':'locality'}, axis=1, inplace=True)
    food.locality.fillna("Ikeja", inplace=True)
    
    food['date'] = pd.to_datetime(food['year'].astype(str) + '-' + food['month'].astype(str), format='%Y-%m')
    
    # Rename the 'date' column to 'date_year_month'
    food.rename(columns={'date': 'date_year_month'}, inplace=True)
    
    # Set the 'date' column as the index of the DataFrame
    food.set_index('date_year_month', inplace=True)

    # Ensure 'price' column is of numeric type
    food['price'] = pd.to_numeric(food['price'], errors='coerce')

    # Drop rows with NaN values in 'price'
    food = food.dropna(subset=['price'])
    
    # Resample the data to monthly frequency and calculate the mean price for each month
    monthly_prices = food['price'].resample('M').mean()

    # Drop NaN values if any, before decomposition
    monthly_prices = monthly_prices.dropna()

    # Decompose the time series to observe seasonal trends
    decomposition = seasonal_decompose(monthly_prices, model='additive')
    seasonal = decomposition.seasonal

    # calculate sales
    sales = food.goods_purchased * food.price
    
    # df.rename({'sales':'sales (€)'}, axis=1, inplace=True)
    # add a new column to the dataframe
    food['sales'] = sales
    return food

food = load_data()

#data structure
st.title("Food Price App")
st.sidebar.title("Filters")

#dataset preview
st.subheader("Data Preview")
st.dataframe(food.head())

# create a filter for country and ticket number
countries =  food['country'].unique()
gpNos10 = food['goods_purchased'].value_counts().head(10).reset_index()["goods_purchased"]

# create a multiselect for countries
selected_countries = st.sidebar.multiselect("country",countries,[countries[0],countries[20]])
top10_gpNos = st.sidebar.selectbox("Top 10 countries",gpNos10[:10])

# filter
filtered_countries = food[food['country'].isin(selected_countries)]

# display the filtered table
st.subheader('Filtered Table')
if not selected_countries:
    st.error('Select a country')
else:
    st.dataframe(filtered_countries.sample(6))


# calculations
totalSales = round(food['sales'].sum())
total_gp = round(food['goods_purchased'].sum())
no_countries = len(countries)
no_filtered_countries = filtered_countries['country'].nunique()
total_filtered_sales = round(filtered_countries['sales'].sum())
total_filtered_gp = round(filtered_countries['goods_purchased'].sum())

# display in columns
col1, col2, col3 = st.columns(3)
if not selected_countries:
    col1.metric('Total Sales', f'${totalSales:,}')
else:
    col1.metric('Total Sales', f'${total_filtered_sales:,}')

if not selected_countries:
    col2.metric('Total goods_purchased', f'{total_gp:,}')
else:
    col2.metric('Total goods_purchased', f'{total_filtered_gp:,}')

# show number of articles based on the filter
if not selected_countries:
    col3.metric('Number of Country', no_countries)
else:
    col3.metric('No. of Countries', no_filtered_countries)


# CHARTS
st.header("Plotting")

# Filter the dataset for the past decade

food_past_decade = food[food.index.year >= 2011]

# Group by year and item, then calculate the average price
average_prices = food_past_decade.groupby(['year', 'produce'])['price'].mean().unstack()

selected_produce = ['Rice - Retail', 'Bread - Retail', 'Milk - Retail', 'Bread - Retail']  # Adjusted based on the dataset
average_prices[selected_produce].plot(figsize=(12, 8))
plt.title('Trends in Global Food Prices Over the Past Decade')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.legend(title='Produce')
plt.grid(True)
plt.show()

# data
countries_grp = food.groupby('country')['sales'].sum()
countries_grp = countries_grp.sort_values(ascending=False)[:-3]
Table= countries_grp.reset_index()

# selection from the filter
filtered_table = Table[Table['country'].isin(selected_countries)]

# Resample the data to monthly frequency and calculate the mean price for each month

monthly_prices = food['price'].resample('M').mean()

# Plot the data using a line plot

plt.figure(figsize=(10, 6))
monthly_prices.plot()
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.title('Seasonal Trends in Food Prices')
plt.grid(True)
plt.show()




# Group data by country and calculate total sales for each country
total_sales_by_country = food.groupby('country')['sales'].sum().sort_values(ascending=False)

# Plot the data using a line plot

st.subheader('Seasonal Trends in Food Prices')
fig0, ax0 = plt.subplots(figsize=(10, 6))
monthly_prices.plot(kind='line')
ax0.set_xlabel('Date')
ax0.set_ylabel('Average Price')
ax0.set_title('Seasonal Trends in Food Prices')
ax0.grid(True)
st.pyplot(fig0)

st.subheader('Country with the highest sales')
fig, ax = plt.subplots(figsize=(12, 6))
total_sales_by_country.plot(kind='bar')
ax.set_xlabel('Date')
ax.set_ylabel('Seasonal Effect on Price')
ax.set_title('Total Sales by Country')
ax.grid(True)
st.pyplot(fig)
# bar plot
st.subheader('Bar Chart')
fig1, ax1 = plt.subplots(figsize=(10,6))
ax1.bar(filtered_table['country'],filtered_table['sales'])
st.pyplot(fig1)

# Pie Chart
# percentages
st.subheader('Pie Chart')
fig2, ax2 = plt.subplots(figsize=(7,5))
ax2.pie(filtered_table['sales'], labels=selected_countries,autopct='%1.1f%%')
st.pyplot(fig2)

st.subheader('Trend Analysis')
daily_sales = food.groupby('date_year_month')['sales'].sum()

fig3, ax3 = plt.subplots(figsize=(12,6))
ax3.plot(daily_sales.index, daily_sales.values)
st.pyplot(fig3)

# st.dataframe(Table)

#st.write(df.head(4))