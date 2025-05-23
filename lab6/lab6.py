import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_points = 100
x = np.random.rand(n_points) * 10  
k_true = 2.0  
b_true = 1.0  
y_true = k_true * x + b_true  
noise = np.random.normal(0, 1, n_points)  
y = y_true + noise  

def least_squares(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    k = numerator / denominator
    b = y_mean - k * x_mean
    return k, b

k_ls, b_ls = least_squares(x, y)

k_poly, b_poly = np.polyfit(x, y, 1)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Дані з шумом')
plt.plot(x, y_true, color='green', label=f'Істинна лінія (y = {k_true}x + {b_true})')
plt.plot(x, k_ls * x + b_ls, color='red', label=f'МНК (y = {k_ls:.2f}x + {b_ls:.2f})')
plt.plot(x, k_poly * x + b_poly, color='orange', label=f'np.polyfit (y = {k_poly:.2f}x + {b_poly:.2f})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Порівняння ліній регресії')
plt.legend()
plt.grid(True)
plt.savefig('regression_comparison.png')
plt.close()

print(f"Істинні параметри: k = {k_true}, b = {b_true}")
print(f"МНК: k = {k_ls:.4f}, b = {b_ls:.4f}")
print(f"np.polyfit: k = {k_poly:.4f}, b = {b_poly:.4f}")

def gradient_descent(x, y, learning_rate=0.01, max_iter=1000, tol=1e-4):
    n = len(x)
    k = 0.0  
    b = 0.0 
    losses = []
    prev_loss = float('inf')

    for i in range(max_iter):
        y_pred = k * x + b
        error = y_pred - y
        dk = (2 / n) * np.sum(error * x)
        db = (2 / n) * np.sum(error)
        k -= learning_rate * dk
        b -= learning_rate * db
        mse = np.mean(error ** 2)
        losses.append(mse)

        if abs(prev_loss - mse) < tol:
            print(f"Зупинка на ітерації {i+1}, MSE = {mse:.4f}")
            break
        prev_loss = mse

    return k, b, losses

learning_rate = 0.01
max_iter = 1000
k_gd, b_gd, losses = gradient_descent(x, y, learning_rate, max_iter)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Дані з шумом')
plt.plot(x, y_true, color='green', label=f'Істинна лінія (y = {k_true}x + {b_true})')
plt.plot(x, k_ls * x + b_ls, color='red', label=f'МНК (y = {k_ls:.2f}x + {b_ls:.2f})')
plt.plot(x, k_poly * x + b_poly, color='orange', label=f'np.polyfit (y = {k_poly:.2f}x + {b_poly:.2f})')
plt.plot(x, k_gd * x + b_gd, color='purple', label=f'Градієнтний спуск (y = {k_gd:.2f}x + {b_gd:.2f})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Порівняння ліній регресії')
plt.legend()
plt.grid(True)
plt.savefig('C:\\Users\\User\\OneDrive\\Desktop\\regression_comparison_with_gd.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(range(len(losses)), losses, color='purple')
plt.yscale('log')  
plt.xlabel('Кількість ітерацій')
plt.ylabel('MSE (логарифмічна шкала)')
plt.title('Похибка від кількості ітерацій')
plt.grid(True)
plt.savefig('C:\\Users\\User\\OneDrive\\Desktop\\loss_vs_iterations_log.png')
plt.close()

print("Перші 5 значень MSE:")
for i in range(min(5, len(losses))):
    print(f"Ітерація {i+1}: MSE = {losses[i]:.4f}")

print(f"Градієнтний спуск: k = {k_gd:.4f}, b = {b_gd:.4f}")
print(f"Різниця з МНК: k = {abs(k_gd - k_ls):.4f}, b = {abs(b_gd - b_ls):.4f}")