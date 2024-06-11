#!/usr/bin/env python
# coding: utf-8

# In[27]:


# https://finance.yahoo.com/quote/GC%3DF/history/?frequency=1d&period1=1275696000&period2=1717587149
# dùng data scaraper để crawl về
import requests
import pandas as pd
import re


# In[46]:


import re

# Đường dẫn đến file 
file_path = "D:/crawdata/crawl_gold/gold.txt"

# Đọc nội dung file
with open(file_path, 'r') as f:
    data = f.read()

# Biểu thức chính quy để tìm tất cả các ngày tháng
dates = re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, \d{4}\b', data)

# Kiểm tra xem có ngày tháng nào được tìm thấy không
if not dates:
    print("Không tìm thấy ngày tháng nào trong file.")
else:
    # In các ngày tháng đã tách ra
    for date in dates:
        date = date.replace(",", " ")
        print(date)

    # Lưu các ngày tháng vào file mới
    output_file_path = "D:/crawdata/data_gold.txt"  # Lưu vào thư mục hiện tại
    with open(output_file_path, 'w') as f_out:
        for date in dates:
            f_out.write(f"{date}\n")
    
    # Xóa các ngày tháng đã tìm ra khỏi file gốc
    for date in dates:
        data = data.replace(date, '')

    # Ghi nội dung đã chỉnh sửa vào file gốc
    with open(file_path, 'w') as f:
        f.write(data)
        


# In[86]:


file_path = "D:/crawdata/crawl_gold/gold.txt"

# Đọc nội dung từ tệp CSV vào danh sách các dòng
with open(file_path, 'r') as file:
    data = file.read().replace(',', '').replace('ï»¿', '')


# Chia dữ liệu thành các giá trị
values = data.split()


new_lines = [values[i:i+6] for i in range(0, len(values), 6)]

output_file_path = 'D:/crawdata/value.csv'
# In ra các hàng mới đã được tạo
for line in new_lines:
    print(line)

df_value = pd.DataFrame(new_lines)


# In[87]:


print(df_value)


# In[88]:


import pandas as pd

file_path = "D:/crawdata/data_gold.txt"

# mở file để đọc
with open(file_path, 'r') as f:
    content = f.read()
    modified_content = content.replace(', ', "/")

# mở file để viết đè 
with open(file_path, 'w') as f:
    # viết lại những gì đã chỉnh sửa
    f.write(modified_content)
    
data = {'Date': modified_content.split('\n')}
df_date = pd.DataFrame(data)

print(df_date)


# In[96]:


df_date.reset_index(drop=True, inplace=True)
df_value.reset_index(drop=True, inplace=True)

df = pd.concat([df_date, df_value], axis=1)
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj close', 'Volume']
df.drop(3522, inplace=True)

print(df)


# In[97]:


#Tiền xly dlieu
df['Open'] = df['Open'].fillna(0).astype('float')
df['High'] = df['High'].fillna(0).astype('float')
df['Low'] = df['Low'].fillna(0).astype('float')
df['Close'] = df['Close'].fillna(0).astype('float')
df['Adj close'] = df['Adj close'].fillna(0).astype('float')



df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
df['Volume'] = df['Volume'].fillna(0).astype('int64')

print(df.info())


# In[98]:


df.describe()


# In[100]:


df.to_csv('D:/crawdata/crawl_gold/gold.csv', index=False)


# In[2]:


file_path = "D:/crawdata/crawl_gold/gold.csv"

df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'], format='%b %d/%Y')


# In[3]:


df = df.sort_values(by='Date')


# In[4]:


print(df)


# In[6]:


#viết lại file khi chỉnh sửa date time
df.to_csv('D:/crawdata/crawl_gold/gold.csv', index=False)


# In[ ]:




