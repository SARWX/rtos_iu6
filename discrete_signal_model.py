import numpy as np

# Исходные параметры
N = 4                  # Число отсчетов
sigma2 = 1.0            # Дисперсия
b = 1e-2                # Параметр, лежащий в [1e-2, 1e-4]
_lambda = 1.0           # lambda: отношение X_fch к X_fh


# Вычисление параметра x_c
x_c = 2 * np.sqrt(2.31 * np.log10(1 / b))

# Массивы коэффициентов
X_fh = np.zeros(N//2 + 1)
X_fch = np.zeros(N//2 + 1)


# Вычисление X_fh(0)
X_fh[0] = sigma2 * x_c / np.sqrt(np.pi * N)

# Вычисление X_fh(N/2)
X_fh[N//2] = (sigma2 * x_c / np.sqrt(np.pi * N)) * np.exp(-x_c / 4)

# Вычисление X_fh(k) для 1 <= k < N/2
for k in range(1, N//2):
    X_fh[k] = (sigma2 * x_c / np.sqrt(np.pi * N)) * np.exp(- (x_c**2 * k**2) / N**2)


# X_fch = lambda * X_fh
X_fch = _lambda * X_fh


# Вычисление x(i) по формуле
def x_i(i):
    result = X_fch[0]
    for k in range(1, N//2):
        cos_term = X_fch[k] * np.cos(2 * np.pi * k * i / N)
        sin_term = X_fh[k] * np.sin(2 * np.pi * k * i / N)
        result += 2 * (cos_term + sin_term)
    result += X_fch[N//2] * np.cos(np.pi * i)
    return result

# Генерация всех x(i)
x_values = np.array([x_i(i) for i in range(N)])
