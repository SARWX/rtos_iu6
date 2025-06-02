import numpy as np
import matplotlib.pyplot as plt

# Наборы параметров для четырёх прогонов
params = [
    (4, 0.1, 1.0),
    (16, 0.05, 1.0),
    (100, 0.01, 1.0),
    (1000, 0.005, 1.0)
]

for run_num, (N, b, sigma2) in enumerate(params, start=1):
    _lambda = 1.0

    # Вычисление параметра x_c
    x_c = 2 * np.sqrt(2.31 * np.log10(1 / b))

    # Массивы коэффициентов
    X_fh = np.zeros(N // 2 + 1)
    X_fch = np.zeros(N // 2 + 1)

    # Вычисление X_fh
    # X_fh[0] = np.sqrt(sigma2 * x_c / np.sqrt(np.pi * N))
    # X_fh[N // 2] = np.sqrt((sigma2 * x_c / np.sqrt(np.pi * N)) * np.exp(-x_c / 4))
    # for k in range(1, N // 2):
    #     X_fh[k] = np.sqrt((sigma2 * x_c / np.sqrt(np.pi * N)) * np.exp(- (x_c**2 * k**2) / N**2))
    X_fh[0] = (sigma2 * x_c / np.sqrt(np.pi * N))
    X_fh[N // 2] = ((sigma2 * x_c / np.sqrt(np.pi * N)) * np.exp(-x_c / 4))
    for k in range(1, N // 2):
        X_fh[k] = ((sigma2 * x_c / np.sqrt(np.pi * N)) * np.exp(- (x_c**2 * k**2) / N**2))
    # X_fh *= (1 / np.sqrt(2))

    X_fch = _lambda * X_fh

    # Генерация x(i)
    def x_i(i):
        result = X_fch[0]
        for k in range(1, N // 2):
            result += 2 * (X_fch[k] * np.cos(2 * np.pi * k * i / N) + X_fh[k] * np.sin(2 * np.pi * k * i / N))
        result += X_fch[N // 2] * np.cos(np.pi * i)
        return result

    x_values = np.array([x_i(i) for i in range(N)])

    # Теоретическая АКФ
    m_vals = np.arange(N)
    x_c_theory = 2 * np.sqrt(2.3 * np.log10(1 / b))
    Rt = sigma2 * np.exp(- (np.pi**2 * m_vals**2) / x_c_theory**2)

    # Оценка АКФ
    Re = np.zeros_like(m_vals, dtype=float)
    for i, m in enumerate(m_vals):
        acc = 0
        for j in range(N - m):
            acc += x_values[j] * x_values[j + m]
        Re[i] = acc / (N - m)

    # Ошибка
    error = np.abs(Rt - Re)
    avg_abs_error = np.mean(error)

    # Построение двух графиков рядом
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Прогон #{run_num}: N={N}, b={b}, sigma2={sigma2} | Средняя ошибка: {avg_abs_error:.4e}')

    # График x(i)
    ax1.plot(range(N), x_values, marker='o', linestyle='-', color='blue', label='x(i)')
    ax1.set_title('Функция x(i)')
    ax1.set_xlabel('i')
    ax1.set_ylabel('x(i)')
    ax1.grid(True)
    ax1.legend()

    # График Rt(m), Re(m), ошибка
    ax2.plot(m_vals, Rt, label=r'$R_t(m)$ - теор. АКФ', color='green', marker='o')
    ax2.plot(m_vals, Re, label=r'$R_э(m)$ - оценка', linestyle='--', marker='o', color='blue')
    ax2.plot(m_vals, error, label=r'ошибка', linestyle='--', marker='o', color='red')
    ax2.set_title('Сравнение Rt(m) и Re(m)')
    ax2.set_xlabel('m')
    ax2.set_ylabel('R(m)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Оставить место для заголовка
    plt.show()
