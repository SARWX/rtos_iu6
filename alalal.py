import numpy as np
import matplotlib.pyplot as plt

# Функция для вычисления x_c
def calculate_xc(b):
    return 2 * np.sqrt(2.3 * np.log10(1 / b))

# Функция для вычисления спектральных коэффициентов X_φH
def calculate_X_phi_H(xc, N):
    sigma_sq = 1
    K = N // 2
    X = np.zeros(K + 1)
    
    # X_φH(0)
    X[0] = sigma_sq * xc / (np.sqrt(np.pi * N))
    
    # X_φH(k) для k=1,...,K-1
    for k in range(1, K):
        X[k] = sigma_sq * xc / (np.sqrt(np.pi * N)) * np.exp(- (xc**2 * k**2) / (N**2))
    
    # X_φH(K) (если N чётное)
    if K < N:
        X[K] = sigma_sq * xc / (np.sqrt(np.pi * N) * np.exp(xc / 4))
    
    return X

# Функция для генерации сигнала x(i)
def generate_signal(X, N):
    K = len(X) - 1  # Последний элемент — X_φH(N/2)
    x = np.zeros(N)
    
    for i in range(N):
        term0 = X[0]
        terms = 0
        
        # Сумма по k=1,...,K-1
        for k in range(1, K):
            terms += X[k] * (np.cos(2 * np.pi * k * i / N) + np.sin(2 * np.pi * k * i / N))
        
        # Термин для X_φH(N/2)
        if K < N:
            term_last = X[K] * np.cos(np.pi * i)
        else:
            term_last = 0
        
        x[i] = term0 + 2 * terms + term_last
    
    return x

# Функция для вычисления теоретической АКФ
def theoretical_R_T(m, xc):
    sigma_sq = 1
    return sigma_sq * np.exp(- (np.pi ** 2) * m ** 2 / (xc ** 2))

# Функция для вычисления экспериментальной АКФ
def experimental_R_E(x, m):
    N = len(x)
    if m >= N:
        return 0
    return np.sum(x[:N - m] * x[m:]) / (N - m)

# Функция для вычисления погрешности
def calculate_error(R_theor, R_exp):
    return np.abs(R_theor - R_exp)

# Список пар (b, N)
b_n_pairs = [(0.1, 4), (0.1, 40), (0.01, 40), (0.01, 400), (0.001, 400)]

for b, N in b_n_pairs:
    # Вычисляем x_c
    xc = calculate_xc(b)
    
    # Вычисляем спектральные коэффициенты
    X = calculate_X_phi_H(xc, N)
    
    # Генерируем сигнал
    x = generate_signal(X, N)
    
    # Рассчитываем теоретическую и экспериментальную АКФ
    m_values = range(N)  # Все возможные m
    R_theor = [theoretical_R_T(m, xc) for m in m_values]
    R_exp = [experimental_R_E(x, m) for m in m_values]
    
    # Вычисляем погрешность и среднюю погрешность
    error = [calculate_error(rt, re) for rt, re in zip(R_theor, R_exp)]
    mean_error = np.mean(error)
    
    # Построение графиков
    plt.figure(figsize=(16, 8))
    
    # График сигнала
    plt.subplot(2, 2, 1)
    plt.plot(x, marker='o', linestyle='-')
    plt.title(f'Дискретный сигнал (N={N}, b={b})')
    plt.xlabel('Отсчеты')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    
    # График АКФ
    plt.subplot(2, 2, 2)
    plt.plot(R_theor, label='Теоретическая АКФ', color='red')
    plt.plot(R_exp, label='Экспериментальная АКФ', color='blue')
    plt.title(f'АКФ (N={N}, b={b})')
    plt.xlabel('m')
    plt.ylabel('АКФ')
    plt.legend()
    plt.grid(True)
    
    # График погрешности
    plt.subplot(2, 2, 3)
    plt.plot(error, color='green')
    plt.title('Погрешность')
    plt.xlabel('m')
    plt.ylabel('Абсолютная погрешность')
    plt.grid(True)
    
    # График средней погрешности
    plt.subplot(2, 2, 4)
    plt.axhline(mean_error, color='red', linestyle='--', label='Средняя погрешность')
    plt.title('Средняя погрешность')
    plt.xlabel('m')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
