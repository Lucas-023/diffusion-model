import numpy as np
import matplotlib.pyplot as plt

def h_over_p(p):
    p = np.atleast_1d(p)
    res = np.zeros_like(p, dtype=float)
    
    # Filtramos p > 0 para evitar divisão por zero
    # E p < 1 para o logaritmo de (1-p)
    mask = (p > 0) & (p < 1)
    
    # h(p) / p = -(p*log2(p) + (1-p)*log2(1-p)) / p
    # Simplificando: -log2(p) - ((1-p)/p) * log2(1-p)
    res[mask] = -(p[mask] * np.log2(p[mask]) + (1 - p[mask]) * np.log2(1 - p[mask])) / p[mask]
    
    # Para p = 1, h(1) = 0, então h(1)/1 = 0
    res[p == 1] = 0
    
    # Para p próximo de 0, o valor explode para o infinito
    res[p == 0] = np.inf
    return res

# Criando o intervalo (começando levemente acima de 0 para ver a subida)
p_values = np.linspace(0.001, 1, 500)
y_values = h_over_p(p_values)

plt.figure(figsize=(8, 5))
plt.plot(p_values, y_values, color='darkorange', linewidth=2, label=r'$\frac{h(p)}{p}$')

# Estilização
plt.title(r'Gráfico da Razão $\frac{h(p)}{p}$', fontsize=14)
plt.xlabel('Probabilidade $p$', fontsize=12)
plt.ylabel('Valor', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 10) # Limitando o eixo Y para melhor visualização
plt.legend()

plt.show()