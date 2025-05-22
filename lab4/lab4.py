import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
import timeit

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

np.random.seed(42)
data['color'] = np.random.choice(['red', 'white'], size=len(data), p=[0.8, 0.2])

data.loc[np.random.choice(data.index, size=int(0.1 * len(data)), replace=False), 'alcohol'] = np.nan
data.loc[np.random.choice(data.index, size=int(0.1 * len(data)), replace=False), 'pH'] = np.nan

numeric_columns = [col for col in data.columns if col != 'color']
data_numeric = data[numeric_columns].astype(float)  
data_np = data_numeric.to_numpy() 
columns = data_numeric.columns.tolist()
alcohol_idx = columns.index('alcohol')
pH_idx = columns.index('pH')

def handle_missing_pandas(df):
    df_filled = df.copy()
    df_filled['alcohol'] = df_filled['alcohol'].fillna(df_filled['alcohol'].mean())
    df_filled['pH'] = df_filled['pH'].fillna(df_filled['pH'].mean())
    return df_filled

def handle_missing_numpy(arr, col_idx):
    arr_filled = arr.copy()
    valid = ~np.isnan(arr_filled[:, col_idx])
    mean_val = np.nanmean(arr_filled[valid, col_idx])
    arr_filled[~valid, col_idx] = mean_val
    return arr_filled

pandas_time_missing = timeit.timeit(lambda: handle_missing_pandas(data), number=100)
data_filled_pandas = handle_missing_pandas(data)

data_filled_np = data_np.copy()
numpy_time_missing = 0
for col_idx in [alcohol_idx, pH_idx]:
    start_time = timeit.default_timer()
    for _ in range(100):
        data_filled_np = handle_missing_numpy(data_filled_np, col_idx)
    numpy_time_missing += (timeit.default_timer() - start_time) / 100

def normalize_pandas(df, col):
    df_copy = df.copy()
    df_copy[col] = (df_copy[col] - df_copy[col].min()) / (df_copy[col].max() - df_copy[col].min())
    return df_copy

def standardize_pandas(df, col):
    df_copy = df.copy()
    df_copy[col] = (df_copy[col] - df_copy[col].mean()) / df_copy[col].std()
    return df_copy

def normalize_numpy(arr, col_idx):
    arr_copy = arr.copy()
    min_val = np.nanmin(arr_copy[:, col_idx])
    max_val = np.nanmax(arr_copy[:, col_idx])
    arr_copy[:, col_idx] = (arr_copy[:, col_idx] - min_val) / (max_val - min_val)
    return arr_copy

def standardize_numpy(arr, col_idx):
    arr_copy = arr.copy()
    mean_val = np.nanmean(arr_copy[:, col_idx])
    std_val = np.nanstd(arr_copy[:, col_idx])
    arr_copy[:, col_idx] = (arr_copy[:, col_idx] - mean_val) / std_val
    return arr_copy

pandas_time_norm = timeit.timeit(lambda: normalize_pandas(data_filled_pandas, 'alcohol'), number=100)
data_norm_pandas = normalize_pandas(data_filled_pandas, 'alcohol')
pandas_time_std = timeit.timeit(lambda: standardize_pandas(data_filled_pandas, 'alcohol'), number=100)
data_std_pandas = standardize_pandas(data_filled_pandas, 'alcohol')

numpy_time_norm = timeit.timeit(lambda: normalize_numpy(data_filled_np, alcohol_idx), number=100)
data_norm_np = normalize_numpy(data_filled_np, alcohol_idx)
numpy_time_std = timeit.timeit(lambda: standardize_numpy(data_filled_np, alcohol_idx), number=100)
data_std_np = standardize_numpy(data_filled_np, alcohol_idx)

plt.figure(figsize=(8, 6))
plt.hist(data_filled_pandas['alcohol'], bins=10, edgecolor='black')
plt.title('Гістограма вмісту алкоголю')
plt.xlabel('Алкоголь')
plt.ylabel('Кількість')
plt.savefig('C:\\Users\\User\\OneDrive\\Desktop\\alcohol_histogram.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(data_filled_pandas['alcohol'], data_filled_pandas['pH'], alpha=0.5)
plt.title('Алкоголь vs pH')
plt.xlabel('Алкоголь')
plt.ylabel('pH')
plt.savefig('C:\\Users\\User\\OneDrive\\Desktop\\alcohol_vs_pH.png')
plt.close()

pearson_corr, _ = stats.pearsonr(data_filled_pandas['alcohol'], data_filled_pandas['pH'])
spearman_corr, _ = stats.spearmanr(data_filled_pandas['alcohol'], data_filled_pandas['pH'])

def one_hot_encode_pandas(df, col):
    return pd.get_dummies(df, columns=[col], prefix=col)

pandas_time_ohe = timeit.timeit(lambda: one_hot_encode_pandas(data_filled_pandas, 'color'), number=100)
data_ohe_pandas = one_hot_encode_pandas(data_filled_pandas, 'color')

sns.pairplot(data_filled_pandas[['alcohol', 'pH', 'quality']])
plt.savefig('C:\\Users\\User\\OneDrive\\Desktop\\pairplot.png')
plt.close()

X = data_filled_pandas[['alcohol']].values
y = data_filled_pandas['pH'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

models = {
    'Лінійна регресія': LinearRegression(),
    'Ridge регресія': Ridge(alpha=1.0),
    'Lasso регресія': Lasso(alpha=1.0)
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = mse

for name, model in models.items():
    plt.figure(figsize=(10, 6))
    y_pred = model.predict(X_test)
    plt.scatter(X_test, y_test, alpha=0.5, label='Фактичні дані')
    plt.scatter(X_test, y_pred, alpha=0.5, label=f'{name} Прогноз')
    plt.title(f'Результати {name}')
    plt.xlabel('Алкоголь')
    plt.ylabel('pH')
    plt.legend()
    plt.savefig(f'{name.lower().replace(" ", "_")}_results.png')
    plt.close()

print("Результати профілювання (секунди, середнє за 100 запусків):")
print(f"Обробка пропущених значень (Pandas): {pandas_time_missing:.4f}")
print(f"Обробка пропущених значень (NumPy): {numpy_time_missing:.4f}")
print(f"Нормалізація (Pandas): {pandas_time_norm:.4f}")
print(f"Нормалізація (NumPy): {numpy_time_norm:.4f}")
print(f"Стандартизація (Pandas): {pandas_time_std:.4f}")
print(f"Стандартизація (NumPy): {numpy_time_std:.4f}")
print(f"One-Hot Encoding (Pandas): {pandas_time_ohe:.4f}")
print(f"Кореляція Пірсона (alcohol vs pH): {pearson_corr:.4f}")
print(f"Кореляція Спірмена (alcohol vs pH): {spearman_corr:.4f}")
print("Результати MSE для регресійних моделей:")
for name, mse in results.items():
    print(f"{name}: {mse:.4f}")
