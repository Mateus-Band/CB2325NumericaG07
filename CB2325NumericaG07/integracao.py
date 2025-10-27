import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from numpy.polynomial import Polynomial
#simplificações:
pi, sin, cos = sp.pi, sp.sin, sp.cos

def integral_trapezio(function, a, b, n : int, plotar = False) -> float:
    '''
    Calcula a integral definida de uma função usando o método do trapézio.
    Compatível com funções do sympy e funções normais do python.

    Parâmetros:
    function: A função a ser integrada.
    a: Limite inferior da integração.
    b: Limite superior da integração.
    n: Número de subintervalos.

    Retorna:
    A aproximação da integral definida.

    Formula:
        ∫[a,b] f(x) dx ≈ (h/2) * [f(x0) + 2*f(x1) + 2*f(x2) + ... + 2*f(x(n-1)) + f(xn]]
        onde h = (b - a) / n e xi = a + i*h para i = 0, 1, ..., n
    '''
    #Tratamento de erros
    if n < 1:
        raise ValueError("O número de subintervalos (n) deve ser pelo menos 1.")
    
    #Conversão feita por IA.
    #converter SymPy -> callable quando necessário
    if not callable(function):
        if isinstance(function, sp.Lambda):
            # extrai variável e expressão de Lambda
            vars_ = function.variables
            if len(vars_) != 1:
                raise TypeError("sp.Lambda com mais de uma variável não é suportado.")
            sym = vars_[0]
            expr = function.expr
        elif isinstance(function, sp.Expr):
            syms = list(function.free_symbols)
            if len(syms) == 0:
                sym = sp.Symbol('x')  # constante — cria símbolo dummy
                expr = function
            elif len(syms) == 1:
                sym = syms[0]
                expr = function
            else:
                raise TypeError("Expressão SymPy com múltiplas variáveis não é suportada.")
        else:
            raise TypeError("function deve ser callable ou uma expressão/lambda do SymPy.")
        function = sp.lambdify(sym, expr, modules=["math"])

    a = float(a)
    b = float(b)
    h = (b - a) / n
    
    # Cálculo da integral (movido para fora do if/else para calcular sempre)
    def x(i):
        return a + i * h
    
    soma_interna = 0
    for i in range(1, n):
        soma_interna += function(x(i))

    integral = (h/2) * (function(a) + 2 * soma_interna + function(b))

    if plotar:
        # Função interna para plotar
        def plot(f):
            # 1. Pontos para a curva suave
            num_pontos_smooth = 300 # Mais pontos para suavidade
            x_smooth = []
            y_smooth = []
            delta_x_smooth = (b - a) / (num_pontos_smooth - 1)
            for i in range(num_pontos_smooth):
                x_val = a + i * delta_x_smooth
                x_smooth.append(x_val)
                y_smooth.append(f(x_val))

            # 2. Pontos para os vértices dos trapézios
            x_trap = []
            y_trap = []
            for i in range(n + 1):
                x_val = a + i * h
                x_trap.append(x_val)
                y_trap.append(f(x_val))

            # --- Criação do Gráfico ---
            _, ax = plt.subplots(figsize=(10, 6))

            #plot da função
            func_label = 'f(x)'
            ax.plot(x_smooth, y_smooth, label=func_label, color='blue', linewidth=2)

            #plot dos trapézios
            for i in range(n):
                ax.plot([x_trap[i], x_trap[i+1]], [y_trap[i], y_trap[i+1]], color='red', linestyle='-', marker='o', markersize=4)

            #sombreia a área sob os trapézios
            ax.fill_between(x_trap, 0, y_trap, color='red', alpha=0.3, label=f'Área Aproximada ({n} trapézios)')

            ax.set_title('Visualização da Regra do Trapézio')
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.legend()
            ax.grid(True)
            ax.axhline(0, color='black', linewidth=0.5)
            plt.show()

        plot(function)

        #valor calculado da integral para print
        return integral
        
    else:
        return integral

def integral_simpson(f, a, b, n):
    """
    Calcula a integral de f no intervalo [a, b] usando a regra de Simpson 3/8.
    Parâmetros:
        f (function): Função a ser integrada
        a (float): Limite inferior da integração
        b (float): Limite superior da integração
        n (int): Número de subintervalos (deve ser múltiplo de 3)
    Retorna:
        float: Aproximação da integral definida de f de a até b.
    """
    if n % 3 != 0:
        raise ValueError("n deve ser múltiplo de 3")
    h = (b - a) / n
    soma = f(a) + f(b)
    
    for i in range(1, n):
        x = a + i * h
        if i % 3 == 0:
            soma += 2 * f(x)
        else:
            soma += 3 * f(x)
    return (3 * h / 8) * soma

#função a integrar
f = lambda x: x**(3) - 2*x**2 + 1
a, b = -10, 10
n = 3
area = integral_simpson(lambda x: f(x), a, b, n)
print(f"Área ≈ {area:.3f}")

#plotagem
x_plot = np.linspace(a, b, 300)
y_plot = f(x_plot)
xi = np.linspace(a, b, n + 1)
yi = f(xi)
#Aproximação Simpson 3/8 por blocos de 3 subintervalos ---
simpson_y = np.zeros_like(x_plot)
for k in range(0, n, 3):
    x_block = xi[k:k+4]
    y_block = yi[k:k+4]
    # Interpolação cúbica
    p = Polynomial.fit(x_block, y_block, 3) 
    mask = (x_plot >= x_block[0]) & (x_plot <= x_block[-1])
    simpson_y[mask] = p(x_plot[mask])

plt.figure(figsize=(8,5))
plt.plot(x_plot, y_plot, 'b', label='f(x) escolhida')
plt.plot(x_plot, simpson_y, 'orange', label='Aproximação Simpson 3/8', linewidth=2)
plt.fill_between(x_plot, simpson_y, color='skyblue', alpha=0.3)
plt.vlines(xi, 0, yi, colors='gray', linestyles='dashed', alpha=0.7)
plt.scatter(xi, yi, color='red', zorder=5)
plt.title("Integração Numérica - Simpson 3/8 com polinômios cúbicos")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

if __name__ == "__main__":
    print(integral_trapezio.__doc__)
    print(integral_simpson.__doc__)
