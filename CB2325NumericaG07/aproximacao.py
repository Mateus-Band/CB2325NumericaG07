import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# Função de Ajuste Polinomial

def ajuste_polinomial(valores_x:list, valores_y:list, grau:int):
    '''
    Marcela/Jarmando, adicionar docstring da função aqui, por favor.
    
    '''

    # Construção da Matriz de Vandermonde
    matriz_list = []

    for valor in valores_x:
        linha = []
        for i in range(grau + 1):
            linha.append(valor**i)
        matriz_list.append(linha)

    x_matriz = np.array(matriz_list)
    
    # Construção da Matriz dos Valores de y

    y_list = []

    for valor in valores_y:
        y_list.append(valor)

    y_list = np.array(y_list)

    # Construção da Matriz de Parâmetros

    matriz_T = x_matriz.T

    coeficientes_list = np.linalg.solve(matriz_T @ x_matriz, matriz_T @ y_list)

    # Função Polinomial Aproximadora

    x = sp.Symbol("x")
    expr = 0

    for i in range(len(coeficientes_list)):
        expr += coeficientes_list[i]*x**i
    
    print(f"Função Polinomial Aproximadora {expr}")

    # Plotando o Gráfico

    f = sp.lambdify(x, expr, "numpy")
    x_func = np.linspace(min(valores_x), max(valores_x), 200)
    y_func = f(x_func)

    plt.scatter(valores_x, valores_y, color="blue", marker="o", label="Dados Fornecidos")
    plt.plot(x_func, y_func, color="black", linewidth=2, label="Função Aproximadora")

    plt.title("Gráfico dos Dados Fornecidos e da Função Aproximadora")
    plt.xlabel("Eixo x")
    plt.ylabel("Eixo y")
    plt.margins(x=0.1, y=0.1)
    plt.grid(True)
    plt.show()


# Função de Ajuste Senoidal

def ajuste_senoidal(valores_x, valores_y):

    # Plotagem dos dados fornecidos para que o usuário indique o período aproximado percebido na amostra

    plt.scatter(valores_x, valores_y, color="blue", marker="o", label="Dados Fornecidos")
    plt.title("Gráfico para Aproximação do Período")
    plt.xlabel("Eixo x")
    plt.ylabel("Eixo y")
    plt.margins(x=0.1, y=0.1)
    plt.grid(True)
    plt.legend()
    plt.show()

    T_aprox = float(input("Digite o Período Aproximado: "))
    freq_aprox = (2*np.pi) / T_aprox

    # Identificação da frequência de menor erro dentro de um intervalo próximo à frequência aproximada

    freq_list = []

    for i in range(-200, 201):
        freq_list.append(freq_aprox + i/100)
    
    erros = dict()

    for freq in freq_list:

        # Construção da Matriz de Vandermonde

        matriz_list = []

        matriz_sin = []
        matriz_cos = []
        matriz_one = []

        for valor in valores_x:
            linha = []
            linha.append(np.sin(freq*valor))
            matriz_sin.append((np.sin(freq*valor)))
            linha.append(np.cos(freq*valor))
            matriz_cos.append((np.cos(freq*valor)))
            linha.append(1)
            matriz_one.append(1)
            matriz_list.append(linha)

        matriz_sin = np.array(matriz_sin)
        matriz_cos = np.array(matriz_cos)
        matriz_one = np.array(matriz_one)

        x_matriz = np.array(matriz_list)
        
        # Construção da Matriz dos Valores de y

        y_list = np.array(valores_y)

        # Construção da Matriz de Parâmetros

        matriz_T = x_matriz.T

        coeficientes_list = np.linalg.solve(matriz_T @ x_matriz, matriz_T @ y_list)

        a, b, C = coeficientes_list[0], coeficientes_list[1], coeficientes_list[2]

        # Cálculo do Erro

        erro = np.linalg.norm(y_list - (a*matriz_sin + b*matriz_cos + C*matriz_one))**2

        erros[freq] = (erro, (a,b, C))

    erros_ordenados = dict(sorted(erros.items(), key=lambda item: item[1][0]))
    
    freq_final = next(iter(erros_ordenados))

    a, b, C = erros_ordenados[freq_final][1][0], erros_ordenados[freq_final][1][1], erros_ordenados[freq_final][1][2]

    # Função Senoidal Aproximadora

    x = sp.Symbol("x")
    expr = sp.sqrt(a**2 + b**2)*sp.sin(freq_final*x + sp.atan2(b,a)) + C

    print(f"Função Senoidal Aproximadora: {expr}")

    # Plotando o Gráfico

    f = sp.lambdify(x, expr, "numpy")
    x_func = np.linspace(min(valores_x), max(valores_x), 600)
    y_func = f(x_func)

    plt.scatter(valores_x, valores_y, color="blue", marker="o", label="Dados Fornecidos")
    plt.plot(x_func, y_func, color="black", linewidth=2, label="Função Aproximadora")

    plt.title("Gráfico dos Dados Fornecidos e da Função Aproximadora")
    plt.xlabel("Eixo x")
    plt.ylabel("Eixo y")
    plt.margins(x=0.1, y=0.1)
    plt.grid(True)
    plt.legend()
    plt.show()


# Função de Ajuste Múltiplo

def ajuste_multiplo(valores_variaveis:list, valores_z:list):

    # Construção da Matriz de Valores das Variáveis

    xm_temp = np.array(valores_variaveis)
    x_matriz = np.insert(xm_temp, 0, 1, axis=1)
    
    
    # Construção da Matriz dos Valores de z

    z_matriz = np.array(valores_z)

    if x_matriz.shape[0] != len(z_matriz):
        raise ValueError("Número de linhas de X e número de valores de Z não coincidem.")

    # Construção da Matriz de Coeficientes

    matriz_T = x_matriz.T

    coeficientes_list = np.linalg.solve(matriz_T @ x_matriz, matriz_T @ z_matriz)
    coeficientes_list = coeficientes_list.ravel()

    # Função Aproximadora para Regressão Múltipla

    qtd_var = x_matriz.shape[1] - 1
    ind_fin = qtd_var + 1
    x = sp.symbols(f"x1:{ind_fin}")
    expr = coeficientes_list[0]

    for i in range(qtd_var):
        expr += coeficientes_list[i + 1]*x[i]
    
    print(f"Função Aproximadora para Regressão Múltipla: {expr}")