import numpy as np

def x(i, N, X_fch, X_fh):
    result = X_fch[0]
    
    for k in range(1, N // 2):
        cos_term = X_fch[k] * np.cos(2 * np.pi * k * i / N)
        sin_term = X_fh[k] * np.sin(2 * np.pi * k * i / N)
        result += 2 * (cos_term + sin_term)
    
    result += X_fch[N // 2] * np.cos(np.pi * i)
    
    return result

# Пример вызова:
# N = 8
# X_fch = np.array([...])
# X_fh = np.array([...])
# values = [x(i, N, X_fch, X_fh) for i in range(N)]
