import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
def make_regression(df):
    x = df[['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']]
    y = df['Weekly_Sales']
    model = LinearRegression()
    model.fit(x,y)
    return model

month_order = ['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November', 'December']
features = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', None)

table = pd.read_csv('Walmart_Sales.csv')
table['Date'] = pd.to_datetime(table['Date'],dayfirst=True)
table_adj = table.copy()
table_adj['Weekly_Sales'] = 100*table['Weekly_Sales']/table['CPI']
table_adj.drop('CPI',axis=1)

# Unadjusted Walmart Whole
table_whole = table.copy()
table_whole['Date'] = table_whole['Date'].dt.month_name()
data_jan = pd.DataFrame()
data_feb = pd.DataFrame()
data_mar = pd.DataFrame()
data_apr = pd.DataFrame()
data_may = pd.DataFrame()
data_jun = pd.DataFrame()
data_jul = pd.DataFrame()
data_aug = pd.DataFrame()
data_sep = pd.DataFrame()
data_oct = pd.DataFrame()
data_nov = pd.DataFrame()
data_dec = pd.DataFrame()

for i in range(1,46):
    data_store = table_whole[table_whole['Store'] == i]
    monthly_data = data_store.groupby('Date',as_index=False)['Weekly_Sales'].mean()
    monthly_data= pd.merge(monthly_data, data_store.groupby('Date', as_index=False)['Holiday_Flag'].mean(), on = 'Date')
    monthly_data= pd.merge(monthly_data, data_store.groupby('Date', as_index=False)['Temperature'].mean(), on = 'Date')
    monthly_data= pd.merge(monthly_data, data_store.groupby('Date', as_index=False)['Fuel_Price'].mean(), on = 'Date')
    monthly_data= pd.merge(monthly_data, data_store.groupby('Date', as_index=False)['CPI'].mean(), on = 'Date')
    monthly_data= pd.merge(monthly_data, data_store.groupby('Date', as_index=False)['Unemployment'].mean(), on = 'Date')
    data_jan = pd.concat([monthly_data[monthly_data['Date'] == 'January'], data_jan])
    data_feb = pd.concat([monthly_data[monthly_data['Date'] == 'February'], data_feb])
    data_mar = pd.concat([monthly_data[monthly_data['Date'] == 'March'], data_mar])
    data_apr = pd.concat([monthly_data[monthly_data['Date'] == 'April'], data_apr])
    data_may = pd.concat([monthly_data[monthly_data['Date'] == 'May'], data_may])
    data_jun = pd.concat([monthly_data[monthly_data['Date'] == 'June'], data_jun])
    data_jul = pd.concat([monthly_data[monthly_data['Date'] == 'July'], data_jul])
    data_aug = pd.concat([monthly_data[monthly_data['Date'] == 'August'], data_aug])
    data_sep = pd.concat([monthly_data[monthly_data['Date'] == 'September'], data_sep])
    data_oct = pd.concat([monthly_data[monthly_data['Date'] == 'October'], data_oct])
    data_nov = pd.concat([monthly_data[monthly_data['Date'] == 'November'], data_nov])
    data_dec = pd.concat([monthly_data[monthly_data['Date'] == 'December'], data_dec])
model_jan = make_regression(data_jan)
model_feb = make_regression(data_feb)
model_mar = make_regression(data_mar)
model_apr = make_regression(data_apr)
model_may = make_regression(data_may)
model_jun = make_regression(data_jun)
model_jul = make_regression(data_jul)
model_aug = make_regression(data_aug)
model_sep = make_regression(data_sep)
model_oct = make_regression(data_oct)
model_nov = make_regression(data_nov)
model_dec = make_regression(data_dec)
models = [model_jan,model_feb,model_mar,model_apr, model_may,model_jun,model_jul,model_aug,model_sep,model_oct,model_nov,model_dec]
jan_abs_dif = 0
feb_abs_dif = 0
mar_abs_dif = 0
may_abs_dif = 0
jun_abs_dif = 0
jul_abs_dif = 0
aug_abs_dif = 0
sep_abs_dif = 0
oct_abs_dif = 0
nov_abs_dif = 0
dec_abs_dif = 0


for i in range(1, 46):
    data_store = table_whole[table_whole['Store'] == i]
    monthly_data = data_store.groupby('Date',as_index=False)['Weekly_Sales'].mean()
    monthly_data= pd.merge(monthly_data, data_store.groupby('Date', as_index=False)['Holiday_Flag'].mean(), on = 'Date')
    monthly_data= pd.merge(monthly_data, data_store.groupby('Date', as_index=False)['Temperature'].mean(), on = 'Date')
    monthly_data= pd.merge(monthly_data, data_store.groupby('Date', as_index=False)['Fuel_Price'].mean(), on = 'Date')
    monthly_data= pd.merge(monthly_data, data_store.groupby('Date', as_index=False)['CPI'].mean(), on = 'Date')
    monthly_data= pd.merge(monthly_data, data_store.groupby('Date', as_index=False)['Unemployment'].mean(), on = 'Date')
    monthly_data['Date'] = pd.Categorical(monthly_data['Date'], categories=month_order,ordered=True)
    monthly_data = monthly_data.sort_values('Date')

    actual_sales = monthly_data['Weekly_Sales']

    predicted_sales = []

    for month_name, model in zip(month_order, models):
        row = monthly_data[monthly_data['Date'] == month_name]

        x_month = row[features]
        prediction = model.predict(x_month)
        predicted_sales.append(prediction)

    plt.figure(figsize=(16, 8))

    plt.plot(
        month_order,
        actual_sales,
        marker='o',
        label='Actual Weekly Sales'
    )

    plt.plot(
        month_order,
        predicted_sales,
        marker='o',
        label='Predicted Weekly Sales'
    )

    plt.title(f'Store {i}: Actual vs Predicted Average Weekly Sales by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Weekly Sales')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    jan_abs_dif = abs(predicted_sales[0] - actual_sales[0])
    feb_abs_dif = abs(predicted_sales[1] - actual_sales[1])
    mar_abs_dif = abs(predicted_sales[2] - actual_sales[2])
    apr_abs_dif = abs(predicted_sales[3] - actual_sales[3])
    may_abs_dif = abs(predicted_sales[4] - actual_sales[4])
    jun_abs_dif = abs(predicted_sales[5] - actual_sales[5])
    jul_abs_dif = abs(predicted_sales[6] - actual_sales[6])
    aug_abs_dif = abs(predicted_sales[7] - actual_sales[7])
    sep_abs_dif = abs(predicted_sales[8] - actual_sales[8])
    oct_abs_dif = abs(predicted_sales[9] - actual_sales[9])
    nov_abs_dif = abs(predicted_sales[10] - actual_sales[10])
    dec_abs_dif = abs(predicted_sales[11] - actual_sales[11])
jan_abs_dif/=45
feb_abs_dif/=45
mar_abs_dif/=45
apr_abs_dif/=45
may_abs_dif/=45
jun_abs_dif/=45
jul_abs_dif/=45
aug_abs_dif/=45
sep_abs_dif/=45
oct_abs_dif/=45
nov_abs_dif/=45
dec_abs_dif/=45
print(f"Mean Absolute Error: {jan_abs_dif}, {feb_abs_dif}, {mar_abs_dif}, {apr_abs_dif}, {may_abs_dif}, {jun_abs_dif}, {jul_abs_dif}, {aug_abs_dif}, {sep_abs_dif}, {oct_abs_dif}, {nov_abs_dif}, {dec_abs_dif}")


# Adjusted Walmart Whole
#
# table_whole_adj = table_adj.copy()
# correlation_adj = table_adj[['Weekly_Sales','Holiday_Flag', 'Temperature', 'Fuel_Price', 'Unemployment']].corr()
# print(correlation_adj.iloc[:,1:].head(1))
#
# x_adj = table_whole[['Holiday_Flag', 'Temperature', 'Fuel_Price', 'Unemployment']]
# y_adj = table_whole['Weekly_Sales']
# x_train_adj, x_test_adj, y_train_adj, y_test_adj = train_test_split(
#     x_adj, y_adj, test_size=0.2, random_state=1
# )
# model_adj = LinearRegression()
# model_adj.fit(x_train_adj, y_train_adj)
# y_pred_adj = model_adj.predict(x_test_adj)
# print("R² score:", r2_score(y_test_adj, y_pred_adj))
# print("Mean Absolute Error:", mean_absolute_error(y_test_adj, y_pred_adj))
#

# Individual Store unadjusted

table['Date'] = pd.to_datetime(table['Date'],dayfirst=True)
# Pivot: rows = weeks, columns = stores, values = sales
store_sales = table.pivot(
    index='Date',
    columns='Store',
    values='Weekly_Sales'
)
plt.figure(figsize=(40, 8))

for store in store_sales.columns:
    plt.plot(
        store_sales.index,
        store_sales[store],
        label=f"Store {store}",
        alpha=0.7
    )

plt.title("Weekly Sales by Store")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend(title="Store Num", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.show()

# Individual Store unadjusted


# Pivot: rows = weeks, columns = stores, values = sales
store_sales_adj = table_adj.pivot(
    index='Date',
    columns='Store',
    values='Weekly_Sales'
)
plt.figure(figsize=(40, 8))

for store in store_sales.columns:
    plt.plot(
        store_sales_adj.index,
        store_sales_adj[store],
        label=f"Store {store}",
        alpha=0.7
    )

plt.title("Weekly Sales by Store Adjusted")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend(title="Store Num", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.show()