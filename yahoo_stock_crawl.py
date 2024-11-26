import requests
from bs4 import BeautifulSoup
import pandas as pd

# Step 1: Most Active Stocks 페이지에서 상위 10개 주식 심볼 추출
most_active_url = "https://finance.yahoo.com/most-active"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Referer": "https://finance.yahoo.com"
}
response = requests.get(most_active_url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# 테이블에서 심볼(Symbol) 추출
symbols = []
table = soup.find("table")
rows = table.find_all("tr")[1:11]  # 상위 10개 행만 추출
for row in rows:
    symbol = row.find("td").text.strip()  # 첫 번째 열이 Symbol
    symbols.append(symbol)

print(f"Top 10 Most Active Symbols: {symbols}")

# Step 2: 각 심볼의 주식 데이터 크롤링
stock_data = []

for symbol in symbols:
    stock_url = f"https://finance.yahoo.com/quote/{symbol}/history"
    response = requests.get(stock_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 데이터 테이블 찾기
    table = soup.find("table", {"class": "table yf-j5d1ld noDl"})
    rows = table.find_all("tr")

    # 데이터 저장
    for row in rows[1:]:
        cols = row.find_all("td")
        if len(cols) > 1:  # 유효한 데이터 행만 처리
            stock_data.append([symbol] + [col.text.strip() for col in cols])

# Step 3: DataFrame 생성 및 저장
columns = ["Symbol", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
df = pd.DataFrame(stock_data, columns=columns)

# 데이터 정제
def clean_numeric_column(column):
    return column.str.replace(",", "").str.extract(r'([\d\.]+)')[0].astype(float, errors='ignore')

df["Open"] = clean_numeric_column(df["Open"])
df["High"] = clean_numeric_column(df["High"])
df["Low"] = clean_numeric_column(df["Low"])
df["Close"] = clean_numeric_column(df["Close"])
df["Adj Close"] = clean_numeric_column(df["Adj Close"])
df["Volume"] = df["Volume"].str.replace(",", "").str.extract(r'([\d]+)')[0].astype(float, errors='ignore')

df.dropna(inplace=True)

# CSV 파일로 저장
output_path = "Top_10_Most_Active_Stocks_Yahoo.csv"
df.to_csv(output_path, index=False)

print(f"Top 10 most active stocks data saved to {output_path}")
