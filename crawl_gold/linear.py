#!/usr/bin/env python
# coding: utf-8

# In[244]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


# In[3]:


data = 'D:/crawldata/crawl_gold/gold.csv'

df = pd.read_csv(data)
df.head(10)


# In[4]:


df.info()


# In[5]:


df.describe()


# In[10]:


sns.pairplot(df[['Open', 'High', 'Close', 'Adj close', 'Volume']])
plt.show()


# In[82]:


sns.pairplot(df[['Open', 'High','Low', 'Close']])
plt.show()


# In[83]:


#chuẩn bị dữ liệu 
X = df[['Open', 'High', 'Low']]
y = df['Close'] #dự đoán giá vàng đóng cửa


# In[84]:


#chia dữ liệu để kiểm tra với 70% dữ liệu học 30% dữ liệu kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[85]:


print('X train: \n')
print(X_train)
print('----------------------')
print('X test: \n')
print(X_test)
print("----------------------")

print('\n ')
print("y train: " )
print(y_train)
print("----------------------")
print("\n y test: ")
print(y_test)


# In[86]:


#xây dựng mô hình và đánh giá trự quan
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)


# In[110]:


print(f'Sai số trung bình: {lr_mse}')


# In[88]:


plt.scatter(y_test, lr_predictions)
plt.xlabel("Giá trị thực tế")
plt.ylabel("Giá trị dự đoán")
plt.title("Thực tế với Dự đoán")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  
plt.show()



# In[89]:


# Lấy giá trị của ngày cuối cùng
last_row = df.iloc[-1]
print("Dòng cuối cùng trong bộ dữ liệu :\n", last_row)


# In[101]:


# Tạo DataFrame cho 1 ngày tiếp theo dựa trên giá trị của ngày cuối cùng
df['Date'] = pd.to_datetime(df['Date'])
# Tạo DataFrame cho 1 ngày tiếp theo dựa trên giá trị của ngày cuối cùng
next_day_feature = pd.DataFrame([last_row])

# Cập nhật giá trị của cột 'Date' cho các ngày tiếp theo bằng cách thêm một ngày
one_day = pd.Timedelta(days=1)
next_day_feature['Date'] = df['Date'].iloc[-1] + one_day
# Dự đoán giá vàng cho 1 ngày tiếp theo bằng cách sử dụng mô hình đã huấn luyện
next_day_prediction = lr_model.predict(next_day_features[['Open', 'High', 'Low']])
print("Dự đoán giá vàng ngày tiếp theo: ")
print(next_day_prediction)


# In[ ]:


# https://finance.yahoo.com/quote/GC%3DF/history/


# # Model 2

# In[103]:


#chuẩn bị dữ liệu 
X1 = df[['Open', 'High', 'Low']]
y1 = df['Close'] #dự đoán giá vàng đóng cửa


# In[227]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.15, random_state=45)


# In[228]:


print('X train: \n')
print(X_train1)
print('----------------------')
print('X test: \n')
print(X_test1)
print("----------------------")

print('\n ')
print("y train: " )
print(y_train1)
print("----------------------")
print("\n y test: ")
print(y_test1)


# In[229]:


lr_predictions1 = lr_model.predict(X_test1)


# In[230]:


#xây dựng mô hình và đánh giá trực quan
lr_model1 = LinearRegression()
lr_model1.fit(X_train1, y_train1)
lr_predictions1 = lr_model.predict(X_test1)
lr_mse1 = mean_squared_error(y_test1, lr_predictions1)


# In[231]:


print(f'sai số trung bình:  {lr_mse1}')


# In[232]:


plt.scatter(y_test1, lr_predictions1)
plt.xlabel("Giá trị thực tế")
plt.ylabel("Giá trị dự đoán")
plt.title("Thực tế với Dự đoán")
plt.plot([min(y_test1), max(y_test1)], [min(y_test1), max(y_test1)], color='red')  
plt.show()


# In[233]:


# Lấy giá trị của ngày cuối cùng
last_row1 = df.iloc[-1]
print("Dòng cuối cùng trong bộ dữ liệu :\n", last_row1)


# In[242]:


df['Date'] = pd.to_datetime(df['Date'])

# Tạo DataFrame cho 1 ngày tiếp theo dựa trên giá trị của ngày cuối cùng trong df
last_row1 = df.iloc[-1]
next_day_feature1 = pd.DataFrame([last_row1])

# Cập nhật giá trị của cột 'Date' cho ngày tiếp theo bằng cách thêm một ngày
one_day = pd.Timedelta(days=1)
next_day_feature1['Date'] = last_row1['Date'] + one_day

# Dự đoán giá vàng cho 1 ngày tiếp theo bằng cách sử dụng mô hình đã huấn luyện
next_day_prediction1 = lr_model1.predict(next_day_feature1[['Open', 'High', 'Low']])

# In kết quả dự đoán
print("Dự đoán giá vàng ngày tiếp theo:")
print(next_day_prediction1)


# In[246]:


#lưu model
joblib.dump(lr_model1, 'lr_model1.pkl')


# In[ ]:




