print("Скрипт почав виконуватися")
import numpy as np
import pandas as pd
import timeit
print("Бібліотеки імпортовані")

print("Завантаження датасету...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')
print("Датасет завантажено")

data = data.dropna()
print("Пропущені значення видалено")

numeric_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
data_np = data[numeric_columns].to_numpy().astype(float)
print("NumPy масив створено")

try:
    def task1_pandas(df):
        return df[df['fixed acidity'] > 8]

    def task1_numpy(arr):
        return arr[arr[:, 0] > 8]

    pandas_time_task1 = timeit.timeit(lambda: task1_pandas(data), number=10)
    numpy_time_task1 = timeit.timeit(lambda: task1_numpy(data_np), number=10)
    result1_pandas = task1_pandas(data)
    result1_numpy = task1_numpy(data_np)

    def task2_pandas(df):
        return df[df['pH'] > 3.5]

    def task2_numpy(arr):
        return arr[arr[:, 8] > 3.5]

    pandas_time_task2 = timeit.timeit(lambda: task2_pandas(data), number=10)
    numpy_time_task2 = timeit.timeit(lambda: task2_numpy(data_np), number=10)
    result2_pandas = task2_pandas(data)
    result2_numpy = task2_numpy(data_np)

    def task3_pandas(df):
        alcohol_range = df[(df['alcohol'] >= 10) & (df['alcohol'] <= 11)]
        return alcohol_range[alcohol_range['free sulfur dioxide'] > alcohol_range['total sulfur dioxide']]

    def task3_numpy(arr):
        alcohol_mask = (arr[:, 10] >= 10) & (arr[:, 10] <= 11)
        alcohol_data = arr[alcohol_mask]
        return alcohol_data[alcohol_data[:, 5] > alcohol_data[:, 6]]

    pandas_time_task3 = timeit.timeit(lambda: task3_pandas(data), number=10)
    numpy_time_task3 = timeit.timeit(lambda: task3_numpy(data_np), number=10)
    result3_pandas = task3_pandas(data)
    result3_numpy = task3_numpy(data_np)

    def task4_pandas(df):
        sample_size = min(500, len(df))
        sample = df.sample(n=sample_size, replace=False)
        return sample[['free sulfur dioxide', 'total sulfur dioxide', 'alcohol']].mean()

    def task4_numpy(arr):
        sample_size = min(500, arr.shape[0])
        indices = np.random.choice(arr.shape[0], sample_size, replace=False)
        sample = arr[indices]
        return np.mean(sample[:, [5, 6, 10]], axis=0)

    pandas_time_task4 = timeit.timeit(lambda: task4_pandas(data), number=10)
    numpy_time_task4 = timeit.timeit(lambda: task4_numpy(data_np), number=10)
    result4_pandas = task4_pandas(data)
    result4_numpy = task4_numpy(data_np)

    def task5_pandas(df):
        high_alcohol = df[df['alcohol'] > 10]
        sulphates_max = high_alcohol[(high_alcohol['sulphates'] > high_alcohol['chlorides']) & 
                                     (high_alcohol['sulphates'] > high_alcohol['residual sugar'])]
        half_len = len(sulphates_max) // 2
        first_half = sulphates_max.iloc[:half_len]
        second_half = sulphates_max.iloc[half_len:]
        return pd.concat([first_half.iloc[::3], second_half.iloc[::4]])

    def task5_numpy(arr):
        high_alcohol_mask = arr[:, 10] > 10
        high_alcohol_data = arr[high_alcohol_mask]
        sulphates_max_mask = (high_alcohol_data[:, 9] > high_alcohol_data[:, 4]) & (high_alcohol_data[:, 9] > high_alcohol_data[:, 3])
        sulphates_max_data = high_alcohol_data[sulphates_max_mask]
        half_len = len(sulphates_max_data) // 2
        first_half = sulphates_max_data[:half_len]
        second_half = sulphates_max_data[half_len:]
        result = np.concatenate([first_half[::3], second_half[::4]])
        return result

    pandas_time_task5 = timeit.timeit(lambda: task5_pandas(data), number=10)
    numpy_time_task5 = timeit.timeit(lambda: task5_numpy(data_np), number=10)
    result5_pandas = task5_pandas(data)
    result5_numpy = task5_numpy(data_np)

    print("Результати профілювання (секунди, середнє за 10 запусків):")
    print(f"Завдання 1 (Pandas): {pandas_time_task1:.4f}")
    print(f"Завдання 1 (NumPy): {numpy_time_task1:.4f}")
    print(f"Завдання 2 (Pandas): {pandas_time_task2:.4f}")
    print(f"Завдання 2 (NumPy): {numpy_time_task2:.4f}")
    print(f"Завдання 3 (Pandas): {pandas_time_task3:.4f}")
    print(f"Завдання 3 (NumPy): {numpy_time_task3:.4f}")
    print(f"Завдання 4 (Pandas): {pandas_time_task4:.4f}")
    print(f"Завдання 4 (NumPy): {numpy_time_task4:.4f}")
    print(f"Завдання 5 (Pandas): {pandas_time_task5:.4f}")
    print(f"Завдання 5 (NumPy): {numpy_time_task5:.4f}")

except Exception as e:
    print(f"Виникла помилка: {e}")
    raise

input("Натисніть Enter для завершення...")