import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from functools import reduce
import math

# Função de Ajuste Linear

def ajuste_linear(valores_x:list, valores_y:list, plt_grafico: bool = True):
    '''
    Calcula o ajuste linear y = ax + b para os dados (valores_x, valores_y) pelo Método dos Mínimos Quadrados (MMQ).

    Além disso, a função exibe um gráfico de dispersão dos pontos e da reta de ajuste.

    Argumentos:
        valores_x (list): Lista de valores da variável independente.
        valores_y (list): Lista de valores da variável dependente.
        plt_grafico (bool, opcional): True (padrão) se o gráfico deve ser plotado, False caso contrário

    Retorna:
        tuple: (a, b), contendo o coeficiente angular (a) e o coeficiente linear (b) da reta de ajuste.
    '''

    # Cálculo do valor médio de x e y

    x_medio = reduce(lambda x, y: x + y, valores_x)/len(valores_x)
    y_medio = reduce(lambda x, y: x + y, valores_y)/len(valores_y)

    # Cálculo da covariância de x e y e da variância de x para cálculo do coeficiente angular

    cov_xy = 0
    var_x = 0

    for i in range(0, len(valores_x)):
        cov_xy += (valores_x[i] - x_medio)*(valores_y[i] - y_medio)
        var_x += (valores_x[i] - x_medio)**2

    # Cálculo do coeficientes

    a = cov_xy/var_x            # Angular
    b = y_medio - a*x_medio     # Independente

    # Plot do gráfico

    if plt_grafico:
        x_func = np.linspace(min(valores_x), max(valores_x), 200)
        y_func = a*x_func + b

        plt.scatter(valores_x, valores_y, color="blue", marker="o", label="Dados Fornecidos")
        plt.plot(x_func, y_func, color="black", linewidth=2, label="Reta de Ajuste Linear")

        plt.title("Gráfico do Ajuste Linear")
        plt.xlabel("Eixo x")
        plt.ylabel("Eixo y")
        plt.margins(x=0.1, y=0.1)
        plt.grid(True)
        plt.legend()
        plt.show()

    return a, b


# Função de Ajuste Polinomial

def ajuste_polinomial(valores_x: list, valores_y: list, grau_pol: int, plt_grafico: bool = True, expr: bool = True):
    '''
    Realiza o ajuste polinomial de grau especificado (grau_pol) para os dados (valores_x, valores_y) pelo Método dos Mínimos Quadrados (MMQ).

    A função monta e resolve o sistema (XᵀX)c = Xᵀy para c, obtendo os coeficientes do polinômio que mais se aproxima dos dados pelo MMQ. 
    Opcionalmente, a função também exibe um gráfico de dispersão dos pontos com o polinômio de ajuste e a forma simbólica da expressão polinomial resultante (func_aprox).

    Argumentos:
        valores_x (list): Lista de valores da variável independente.
        valores_y (list): Lista de valores da variável dependente.
        grau_pol (int): Grau do polinômio ao qual os dados serão ajustados.
        plt_grafico (bool, opcional): Se True (padrão), exibe o gráfico de ajuste; se False, não exibe.
        expr (bool, opcional): Se True (padrão), exibe a função simbólica do polinômio aproximador; se False, não exibe.

    Retorna:
        list: lista_coeficientes, contendo os coeficientes do polinômio em ordem crescente do grau da variável associada.
    '''

    if len(valores_x) != len(valores_y):
        raise ValueError("As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho.")

    # Construir matriz de Vandermonde (x_matriz)

    x_matriz = np.array([[valor**i for i in range(grau_pol + 1)] for valor in valores_x])
    
    # Construir a matriz dos valores de y (y_matriz)

    y_matriz = np.array(valores_y)

    # Construir a matriz de parâmetros (lista_coeficientes)

    matriz_T = x_matriz.T

    lista_coeficientes = np.linalg.solve(matriz_T @ x_matriz, matriz_T @ y_matriz)

    # Gerar função polinomial aproximadora simbólica (func_aprox)

    x_sym = sp.Symbol("x")
    func_aprox = 0

    for i in range(len(lista_coeficientes)):
        func_aprox += lista_coeficientes[i]*x_sym**i

    if expr:
        print(f"Função Polinomial Aproximadora: {func_aprox}")

    # Plotar o gráfico

    if plt_grafico:

        f = sp.lambdify(x_sym, func_aprox, "numpy")
        x_func = np.linspace(min(valores_x), max(valores_x), 200)
        y_func = f(x_func)

        plt.scatter(valores_x, valores_y, color="blue", marker="o", label="Dados Fornecidos")
        plt.plot(x_func, y_func, color="black", linewidth=2, label="Função Aproximadora")

        plt.title(f"Gráfico dos Dados Fornecidos e da Função Polinomial Aproximadora de grau {grau_pol}")
        plt.xlabel("Eixo x")
        plt.ylabel("Eixo y")
        plt.margins(x=0.1, y=0.1)
        plt.grid(True)
        plt.legend()
        plt.show()
    
    # Retornar os coeficientes (lista_coeficientes)

    return lista_coeficientes


# Função de Ajuste Senoidal

def ajuste_senoidal(valores_x, valores_y, plt_grafico: bool = True, expr: bool = True):
    '''
    Realiza o ajuste senoidal para os dados (valores_x, valores_y) pelo Método dos Mínimos Quadrados (MMQ).

    Modelo adotado:
        y = A * sin(B*x + D) + C. 

    Forma linearizada:
        y = a * sin(B*x) + b * cos(B*x) + c
    onde a = A * cos(D), b = A * sin(D), c = C.

    A função estima a frequência B inicialmente pela aproximação do período fornecida pelo usuário. 
    Em seguida, ela testa diversas frequências em torno da frequência inicial ao resolver o sistema (XᵀX)k = Xᵀy 
    (k = [a, b, c]) para cada uma delas.
    Por fim, ao encontrar a frequência que resulta no menor erro quadrático, ela gera a lista de coeficientes A, B, C e D da função senoidal aproximadora.

    Opcionalmente, a função também exibe um gráfico de dispersão dos pontos com a senóide de ajuste 
    e a forma simbólica da expressão senoidal resultante (func_aprox).

    Argumentos:
        valores_x (list): Lista de valores da variável independente.
        valores_y (list): Lista de valores da variável dependente.
        plt_grafico (bool, opcional): Se True (padrão), exibe o gráfico de ajuste; se False, não exibe.
        expr (bool, opcional): Se True (padrão), exibe a função senoidal simbólica aproximadora; se False, não exibe.

    Retorna:
        list: lista_coeficientes, contendo os coeficientes A, B, C e D.
    '''

    if len(valores_x) != len(valores_y):
        raise ValueError("As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho.")

    # Plotar o gráfico de dispersão dos dados fornecidos para que o usuário indique o período aproximado percebido na amostra

    plt.scatter(valores_x, valores_y, color="blue", marker="o", label="Dados Fornecidos")
    plt.title("Gráfico para Aproximação do Período")
    plt.xlabel("Eixo x")
    plt.ylabel("Eixo y")
    plt.margins(x=0.1, y=0.1)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Captar o período aproximado e cálculo da frequência aproximada 

    T_aprox = float(input("Digite o Período Aproximado: "))
    freq_aprox = (2*np.pi) / T_aprox

    # Identificar a frequência de menor erro quadrático dentro de um intervalo próximo à frequência aproximada
    # São testadas 400 frequências igualmente espaçadas em torno da frequência inicial
    # Para cada frequência, resolve-se o sistema (XᵀX)k = Xᵀy e calcula-se o erro total, de modo que a frequência de menor erro possa ser escolhida

    freq_list = np.linspace(freq_aprox - 2, freq_aprox + 2, 400)
    erros = dict()

    for freq in freq_list:

        # Construir a matriz X das funções base (x_matriz) no formato [sin(Bx), cos(Bx), 1]

        x_matriz = np.array([[np.sin(freq*valor), np.cos(freq*valor), 1] for valor in valores_x])

        matriz_sin = np.array([np.sin(freq*valor) for valor in valores_x])
        matriz_cos = np.array([np.cos(freq*valor) for valor in valores_x])
        matriz_one = np.array([1 for i in range(len(valores_x))])
        
        # Construir a matriz dos valores de y (y_matriz)

        y_matriz = np.array(valores_y)

        # Construir a matriz de parâmetros iniciais (lista_coeficientes), ou seja: a, b e c

        matriz_T = x_matriz.T

        try:
            coeff_iniciais = np.linalg.solve(matriz_T @ x_matriz, matriz_T @ y_matriz)
        except np.linalg.LinAlgError:
            continue  # pula frequências com sistema singular

        a, b, c = coeff_iniciais[0], coeff_iniciais[1], coeff_iniciais[2]

        # Calcular o erro para a frequência em questão

        erro = np.linalg.norm(y_matriz - (a*matriz_sin + b*matriz_cos + c*matriz_one))**2

        erros[freq] = (erro, (a, b, c))
    
    # Obter a frequência de menor erro (freq_final) e seus parâmetros (a, b e c)

    erros_ordenados = dict(sorted(erros.items(), key=lambda item: item[1][0]))
    freq_final = next(iter(erros_ordenados))

    a, b, c = erros_ordenados[freq_final][1][0], erros_ordenados[freq_final][1][1], erros_ordenados[freq_final][1][2]

    # Gerar lista de coeficientes (lista_coeficientes)
    # Contém A, B, C, D tal que a função aproximadora é definida como y = A * sin(B*x + D) + C

    lista_coeficientes = [sp.sqrt(a**2 + b**2), freq_final, c, sp.atan2(b,a)]
    A, B, C, D = lista_coeficientes[0], lista_coeficientes[1], lista_coeficientes[2], lista_coeficientes[3]

    # Gerar função senoidal aproximadora simbólica (func_aprox)

    x_sym = sp.Symbol("x")
    func_aprox = A*sp.sin(B*x_sym + D) + C

    if expr:
        print(f"Função Senoidal Aproximadora: {func_aprox}")

    # Plotar o Gráfico

    if plt_grafico:
        f = sp.lambdify(x_sym, func_aprox, "numpy")
        x_func = np.linspace(min(valores_x), max(valores_x), 600)
        y_func = f(x_func)

        plt.scatter(valores_x, valores_y, color="blue", marker="o", label="Dados Fornecidos")
        plt.plot(x_func, y_func, color="black", linewidth=2, label="Função Aproximadora")

        plt.title("Gráfico dos Dados Fornecidos e da Função Senoidal Aproximadora")
        plt.xlabel("Eixo x")
        plt.ylabel("Eixo y")
        plt.margins(x=0.1, y=0.1)
        plt.grid(True)
        plt.legend()
        plt.show()
    
    # Retornar os coeficientes (lista_coeficientes)

    return lista_coeficientes

# Função de Ajuste Exponencial

def ajuste_exponencial(valores_x:list, valores_y:list, plt_grafico: bool = True):
    '''
    Calcula o ajuste exponencial y = b * e^(a*x) para os dados (valores_x, valores_y) pela linearização do modelo ln(y) = ln(b) + a*x e
    aplica o Método dos Mínimos Quadrados sobre os dados transformados por meio da função de ajuste linear.

    Além disso, a função exibe um gráfico de dispersão dos pontos e da curva de ajuste.

    Argumentos:
        valores_x (list): Lista de valores da variável independente.
        valores_y (list): Lista de valores da variável dependente.
        plt_grafico (bool, opcional): True (padrão) se o gráfico deve ser plotado, False caso contrário

    Retorna:
        tuple: (a, b), contendo o coeficiente do expoente (a) e o coeficiente multiplicativo (b) da curva de ajuste.
    '''

    # Transforma o ajuste exponencial em um ajuste linear

    ln_valores_y = [math.log(y) for y in valores_y]
    a, b_aux = ajuste_linear(valores_x, ln_valores_y, False)
    b = math.exp(b_aux)

    # Plot do gráfico

    if plt_grafico:
        x_func = np.linspace(min(valores_x), max(valores_x), 200)
        y_func = b * np.exp(a * x_func)

        plt.scatter(valores_x, valores_y, color="blue", marker="o", label="Dados Fornecidos")
        plt.plot(x_func, y_func, color="black", linewidth=2, label="Curva de Ajuste Exponencial")

        plt.title("Gráfico do Ajuste Exponencial")
        plt.xlabel("Eixo x")
        plt.ylabel("Eixo y")
        plt.margins(x=0.1, y=0.1)
        plt.grid(True)
        plt.legend()
        plt.show()

    return a, b

# Função de Ajuste Múltiplo

def ajuste_multiplo(valores_variaveis:list, valores_z:list):
    '''
    Marcela/Jarmando, adicionar docstring da função aqui, por favor.
    '''

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