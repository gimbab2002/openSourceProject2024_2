import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
data = pd.read_csv("Top_10_Most_Active_Stocks_Yahoo.csv")
print(data["Symbol"].nunique())  # Symbol 컬럼의 고유값 개수
print(data["Symbol"].unique())  # 모든 고유 Symbol 값

# 1. GroupBy를 활용한 데이터 그룹화 및 통계 분석
grouped = data.groupby("Symbol").agg({
    "Open": ["mean", "max", "min"],
    "Close": ["mean", "max", "min"],
    "Volume": "sum"
})
print(grouped)
print(grouped.index)  # 그룹화된 Symbol 값

# 결과 저장
grouped.to_csv("grouped_data.csv")
print(data.isnull().sum())  # 결측값 확인
print(data.head())  # 데이터의 상위 5개 행 확인

# 2. 데이터 시각화
plt.figure(figsize=(10, 6))
grouped["Volume"]["sum"].plot(kind="bar")
plt.title("Total Volume by Symbol")
plt.xlabel("Symbol")
plt.ylabel("Total Volume")
plt.savefig("volume_by_symbol.png")
plt.show()

plt.figure(figsize=(10, 6))
grouped["Close"]["mean"].plot(kind="line", marker='o')
plt.title("Average Closing Price by Symbol")
plt.xlabel("Symbol")
plt.xticks(range(len(grouped.index)), grouped.index, rotation=45)
plt.ylabel("Average Closing Price")
plt.savefig("average_closing_price.png")
plt.show()

# 3. 주가 변동성 분석 (Close 가격의 표준편차)
volatility = data.groupby("Symbol")["Close"].std()
print(volatility)
volatility.to_csv("volatility.csv")

# 변동성 시각화
plt.figure(figsize=(10, 6))
volatility.plot(kind="bar", color="orange")
plt.title("Volatility by Symbol (StdDev of Close Prices)")
plt.xlabel("Symbol")
plt.ylabel("Volatility")
plt.savefig("volatility_by_symbol.png")
plt.show()

# 4. 상관관계 분석 (Symbol별 주가 피벗)
pivot_data = data.pivot_table(index="Date", columns="Symbol", values="Close")
correlation_matrix = pivot_data.corr()
print(correlation_matrix)
correlation_matrix.to_csv("correlation_matrix.csv")

# 상관관계 시각화 (히트맵)
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="none")
plt.colorbar(label="Correlation Coefficient")
plt.title("Correlation Matrix of Closing Prices")
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.savefig("correlation_matrix.png")
plt.show()

# 7. 일일 수익률 분석
data["DailyReturn"] = data.groupby("Symbol")["Close"].pct_change()
daily_return_mean = data.groupby("Symbol")["DailyReturn"].mean()
print(daily_return_mean)
daily_return_mean.to_csv("daily_return_mean.csv")

# 수익률 시각화
plt.figure(figsize=(10, 6))
daily_return_mean.plot(kind="bar", color="green")
plt.title("Average Daily Return by Symbol")
plt.xlabel("Symbol")
plt.ylabel("Average Daily Return")
plt.savefig("daily_return_by_symbol.png")
plt.show()
