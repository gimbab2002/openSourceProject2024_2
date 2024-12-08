import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 데이터 로드
data = pd.read_csv("Top_10_Most_Active_Stocks_Yahoo.csv")

# 데이터 분석
# 1. GroupBy를 활용한 데이터 그룹화 및 통계 분석
grouped = data.groupby("Symbol").agg({
    "Open": ["mean", "max", "min"],
    "Close": ["mean", "max", "min"],
    "Volume": "sum"
})
print(grouped)
grouped.to_csv("grouped_data.csv")

# 2. 주가 변동성 분석 (Close 가격의 표준편차)
volatility = data.groupby("Symbol")["Close"].std()
volatility.to_csv("volatility.csv")

# 변동성 시각화
plt.figure(figsize=(10, 6))
volatility.plot(kind="bar", color="orange")
plt.title("Volatility by Symbol (StdDev of Close Prices)")
plt.xlabel("Symbol")
plt.ylabel("Volatility")
plt.savefig("volatility_by_symbol.png")
plt.show()

# 3. 상관관계 분석 (Symbol별 주가 피벗)
data["Date"] = pd.to_datetime(data["Date"])  # 날짜를 datetime 형식으로 변환
data = data.sort_values(by=["Date", "Symbol"])  # 날짜 및 Symbol 기준 정렬
pivot_data = data.pivot_table(index="Date", columns="Symbol", values="Close")
correlation_matrix = pivot_data.corr()
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

# 4. 일일 수익률 분석
data["DailyReturn"] = data.groupby("Symbol")["Close"].pct_change()
daily_return_mean = data.groupby("Symbol")["DailyReturn"].mean()
daily_return_mean.to_csv("daily_return_mean.csv")

# 수익률 시각화
plt.figure(figsize=(10, 6))
daily_return_mean.plot(kind="bar", color="green")
plt.title("Average Daily Return by Symbol")
plt.xlabel("Symbol")
plt.ylabel("Average Daily Return")
plt.savefig("daily_return_by_symbol.png")
plt.show()

# 머신러닝: 주가 예측
# 훈련 데이터와 테스트 데이터 분할 (시간 순서를 고려)
train_size = int(len(data) * 0.7)  # 전체 데이터의 70%를 훈련 데이터로 사용
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# 피처와 타겟 변수 설정
X_train = train_data[["Open", "High", "Low", "Volume"]]
y_train = train_data["Close"]

X_test = test_data[["Open", "High", "Low", "Volume"]]
y_test = test_data["Close"]

# 모델 생성 및 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 모델 평가
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 예측 결과를 데이터프레임에 추가
test_data = test_data.copy()
test_data["Predicted"] = y_pred

# 각 Symbol 별로 시각화
unique_symbols = test_data["Symbol"].unique()  # 고유 Symbol 값 추출

for symbol in unique_symbols:
    # 해당 Symbol 데이터 필터링
    symbol_test_data = test_data[test_data["Symbol"] == symbol]
    symbol_dates = symbol_test_data["Date"]
    symbol_y_test = symbol_test_data["Close"]
    symbol_y_pred = symbol_test_data["Predicted"]

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(symbol_dates, symbol_y_test, color="red", label="Actual Prices", alpha=0.7)
    plt.scatter(symbol_dates, symbol_y_pred, color="blue", label="Predicted Prices", alpha=0.7)
    plt.title(f"Actual vs Predicted Prices for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Closing Prices")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"actual_vs_predicted_{symbol}.png")
    plt.show()
