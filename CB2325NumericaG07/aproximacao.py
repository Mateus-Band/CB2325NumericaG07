import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from functools import reduce
import math
import copy

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


# Função de Melhor Ajuste

def melhor_ajuste(valores_x: list, valores_y: list, criterio: str, mostrar_todos: bool = True, plt_grafico: bool = True):

    n = len(valores_x)

    # Obter os parâmetros dos ajustes linear, polinomial (grau 2 a 10), senoidal e exponencial

    funcs = dict()

    funcs["linear"] = {"params": [par for par in ajuste_linear(valores_x, valores_y, plt_grafico=False)]}

    for grau in range(2, 11):
        funcs[f"polinomial grau {grau}"] = {"params": ajuste_polinomial(valores_x, valores_y, grau, plt_grafico=False, expr=False)}
    
    funcs["senoidal"] = {"params": ajuste_senoidal(valores_x, valores_y, plt_grafico=False, expr=False)}

    funcs["exponencial"] = {"params": [par for par in ajuste_exponencial(valores_x, valores_y, plt_grafico=False)]}

    # Calcular SST

    valores_y = np.array(valores_y)
    media_y = np.mean(valores_y)

    copia_y = valores_y.copy()

    SST = np.sum((copia_y - media_y)**2)

    # Ajuste Linear

    # Ajuste Linear - Calcular SSR

    y_lin = np.array(funcs["linear"]["params"][0]*valores_x + funcs["linear"]["params"][1])

    SSR_lin = np.sum((copia_y - y_lin)**2)

    funcs["linear"]["SSR"] = SSR_lin

    # Ajustes Polinomiais

    x_sym = sp.Symbol("x")

    for i in range(2, 11):

        # Ajustes Polinomiais - Calcular SSR

        func_aprox = 0

        for j in range(i):
            func_aprox += funcs[f"polinomial grau {i}"]["params"][j]*x_sym**j

        f_pol = sp.lambdify(x_sym, func_aprox, "numpy")
        y_pol = np.array(f_pol(valores_x))

        # Ajustes Polinomiais - Calcular SSR

        SSR_pol = np.sum((copia_y - y_pol)**2)
        funcs[f"polinomial grau {i}"]["SSR"] = SSR_pol

    # Ajuste Senoidal

    A, B, C, D = funcs["senoidal"]["params"][0], funcs["senoidal"]["params"][1], funcs["senoidal"]["params"][2], funcs["senoidal"]["params"][3]

    ff_sin = A*sp.sin(B*x_sym + D) + C
    f_sin = sp.lambdify(x_sym, ff_sin, "numpy")

    y_sin = np.array(f_sin(valores_x))

    # Ajuste Senoidal - Calcular SSR

    SSR_sin = np.sum((copia_y - y_sin)**2)

    funcs["senoidal"]["SSR"] = SSR_sin

    # Ajuste Exponencial

    a, b = funcs["exponencial"]["params"][0], funcs["exponencial"]["params"][1]

    y_exp = np.array(b*np.exp(a*valores_x))

    # Ajuste Exponencial - Calcular SSR

    SSR_exp = np.sum((copia_y - y_exp)**2)

    funcs["exponencial"]["SSR"] = SSR_exp

    # R^2, R^2 ajustado, AIC e BIC para Ajustes Linear, Senoidal, Exponencial e Polinomiais

    lista_ajustes = ["linear", "senoidal", "exponencial"] + [f"polinomial grau {i}" for i in range(2, 11)]

    for ajuste in lista_ajustes:

        if ajuste == "linear" or ajuste == "exponencial":
            p = 2
        elif ajuste == "senoidal":
            p = 3
        else:
            grau = int(ajuste.split()[-1])
            
            if grau !=0:
                p = grau + 1
            else:
                p = 11
        
        # Calcular R^2

        SSR = funcs[ajuste]["SSR"]

        R2 = 1 - (SSR/SST)
        funcs[ajuste]["R2"] = R2

        # Calcular R^2 Ajustado

        R2A = 1 - ((1 - R2) * (n - 1)/(n - 1 - p))
        funcs[ajuste]["R2A"] = R2A

        # Calcular AIC

        AIC = n * np.log(SSR / n) + 2 * p
        funcs[ajuste]["AIC"] = AIC

        # Calcular BIC

        BIC = n * np.log(SSR / n) + p*np.log(n)
        funcs[ajuste]["BIC"] = BIC
    
    if criterio == "R2":
        funcs_ordenadas = dict(sorted(funcs.items(), key=lambda item: item[1][criterio], reverse=True))
    elif criterio == "R2A":
        funcs_ordenadas = dict(sorted(funcs.items(), key=lambda item: item[1][criterio], reverse=True))
    elif criterio == "AIC":
        funcs_ordenadas = dict(sorted(funcs.items(), key=lambda item: item[1][criterio]))
    elif criterio == "BIC":
        funcs_ordenadas = dict(sorted(funcs.items(), key=lambda item: item[1][criterio]))
    
    aprox_escolhida = next(iter(funcs_ordenadas))
    
    print(f"A sugestão de aproximação para o critério escolhido é {aprox_escolhida}")

    if mostrar_todos:
        print(f"R2: {funcs[aprox_escolhida]['R2']}")
        print(f"R2 Ajustado: {funcs[aprox_escolhida]['R2A']}")
        print(f"AIC: {funcs[aprox_escolhida]['AIC']}")
        print(f"BIC: {funcs[aprox_escolhida]['BIC']}")

    if plt_grafico:
        if aprox_escolhida == "linear":
            ajuste_linear(valores_x, valores_y, plt_grafico=True)
        elif aprox_escolhida == "senoidal":
            ajuste_senoidal(valores_x, valores_y, plt_grafico=True, expr=False)
        elif aprox_escolhida == "exponencial":
            ajuste_exponencial(valores_x, valores_y, plt_grafico=True)
        else:
            if aprox_escolhida[-1] != "0":
                grau = int(aprox_escolhida[-1])
                ajuste_polinomial(valores_x, valores_y, grau, plt_grafico=True, expr=False)
            else:
                ajuste_polinomial(valores_x, valores_y, 10, plt_grafico=True, expr=False)
    
    return aprox_escolhida