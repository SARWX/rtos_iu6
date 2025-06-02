import numpy as np
import matplotlib.pyplot as plt

# Исходные параметры
N = 4                  # Число отсчетов
sigma2 = 1.0            # Дисперсия
# b = 1e-1                # Параметр, лежащий в [1e-2, 1e-4]
b = 0.1
_lambda = 1.0           # lambda: отношение X_fch к X_fh


# Вычисление параметра x_c
x_c = 2 * np.sqrt(2.31 * np.log10(1 / b))

# Массивы коэффициентов
X_fh = np.zeros(N//2 + 1)
X_fch = np.zeros(N//2 + 1)


# Вычисление X_fh(0)
X_fh[0] = np.sqrt(sigma2 * x_c / np.sqrt(np.pi * N))

# Вычисление X_fh(N/2)
X_fh[N//2] = np.sqrt((sigma2 * x_c / np.sqrt(np.pi * N)) * np.exp(-x_c / 4))

# Вычисление X_fh(k) для 1 <= k < N/2
for k in range(1, N//2):
    X_fh[k] = np.sqrt((sigma2 * x_c / np.sqrt(np.pi * N)) * np.exp(- (x_c**2 * k**2) / N**2))


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



# Построение графика
plt.figure(figsize=(10, 4))
plt.plot(range(N), x_values, marker='o', linestyle='-', color='blue', label='x(i)')

plt.title('Функция x(i), восстановленная по спектру')
plt.xlabel('i')
plt.ylabel('x(i)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




# Параметры
m_vals = np.arange(0, N)

# Расчёт x_c и теоретической АКФ Rt(m)
xc = 2 * np.sqrt(2.3 * np.log10(1 / b))
Rt = sigma2 * np.exp(- (np.pi**2 * m_vals**2) / xc**2)

# # Вычисление R_э(m) по формуле (оценка АКФ)
# Re = np.array([
#     tmp = np.sum(x_values[:N - m] * x_values[m:]) / (N - m)
#     for m in m_vals
# ])

Re = np.zeros(len(m_vals))
for m in m_vals:
    acc = 0
    for i in range(N - m):
        acc += x_values[i] * x_values[i + m]
    Re[m] = acc / (N - m)


# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(m_vals, Rt, label=r'$R_t(m)$ — теоретическая АКФ', color='green', marker='o')
plt.plot(m_vals, Re, label=r'$R_э(m)$ — оценка АКФ', linestyle='--', marker='o', color='blue')
plt.plot(m_vals, abs(Rt - Re), label=r'ошибка', linestyle='--', marker='o', color='green')

avg_abs_error = np.mean(np.abs(Rt - Re))

plt.title('Сравнение Rt(m) и Re(m)')
plt.xlabel('m')
plt.ylabel('R(m)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()