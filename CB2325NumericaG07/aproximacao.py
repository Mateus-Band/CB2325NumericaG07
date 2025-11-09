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

    # Verifica se o tamanho da lista dos valores de x e valores de y é igual

    if len(valores_x) != len(valores_y):
        raise ValueError("As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho.")

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

def ajuste_polinomial(valores_x: list[float], valores_y: list[float], grau_pol: int, plt_grafico: bool = True, expr: bool = True) -> np.ndarray:
    """
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
        numpy.ndarray: array_coeficientes, contendo os coeficientes do polinômio em ordem crescente do grau da variável associada.
    """

    if len(valores_x) != len(valores_y): # A quantidade de valores da variável independente deve ser igual à quantidade de valores da variável dependente fornecida
        raise ValueError("As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho.")

    # Construir matriz de Vandermonde (x_matriz)

    x_matriz = np.array([[valor**i for i in range(grau_pol + 1)] for valor in valores_x])
    
    # Construir a matriz dos valores de y (y_matriz)

    y_matriz = np.array(valores_y)

    # Construir a matriz de parâmetros (array_coeficientes)

    matriz_T = x_matriz.T

    array_coeficientes = np.linalg.solve(matriz_T @ x_matriz, matriz_T @ y_matriz)

    # Gerar função polinomial aproximadora simbólica (func_aprox)

    x_sym = sp.Symbol("x")
    func_aprox = 0

    for i in range(len(array_coeficientes)):
        func_aprox += array_coeficientes[i]*x_sym**i

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
    
    # Retornar os coeficientes (array_coeficientes)

    return array_coeficientes

# Função de Ajuste Senoidal

def ajuste_senoidal(valores_x: list[float], valores_y: list[float], plt_grafico: bool = True, expr: bool = True) -> np.ndarray:
    """
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
        numpy.ndarray: array_coeficientes, contendo os coeficientes A, B, C e D.
    """

    if len(valores_x) != len(valores_y): # A quantidade de valores da variável independente deve ser igual à quantidade de valores da variável dependente fornecida
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

    freq_list = np.linspace(freq_aprox * 0.5, freq_aprox * 1.5, 400)
    erros = dict()

    for freq in freq_list:

        # Construir a matriz X das funções base (x_matriz) no formato [sin(Bx), cos(Bx), 1]

        x_matriz = np.array([[np.sin(freq*valor), np.cos(freq*valor), 1] for valor in valores_x])

        matriz_sin = np.array([np.sin(freq*valor) for valor in valores_x])
        matriz_cos = np.array([np.cos(freq*valor) for valor in valores_x])
        
        # Construir a matriz dos valores de y (y_matriz)

        y_matriz = np.array(valores_y)

        # Construir a matriz de parâmetros iniciais (coeff_iniciais), ou seja: a, b e c

        matriz_T = x_matriz.T

        try:
            coeff_iniciais = np.linalg.solve(matriz_T @ x_matriz, matriz_T @ y_matriz)
        except np.linalg.LinAlgError:
            continue  # pula frequências com sistema singular

        a, b, c = coeff_iniciais[0], coeff_iniciais[1], coeff_iniciais[2]

        # Calcular o erro para a frequência em questão

        erro = np.linalg.norm(y_matriz - (a*matriz_sin + b*matriz_cos + c))**2

        erros[freq] = (erro, (a, b, c))
    
    # Obter a frequência de menor erro (freq_final) e seus parâmetros (a, b e c)

    erros_ordenados = dict(sorted(erros.items(), key=lambda item: item[1][0]))
    freq_final = next(iter(erros_ordenados))

    a, b, c = erros_ordenados[freq_final][1][0], erros_ordenados[freq_final][1][1], erros_ordenados[freq_final][1][2]

    # Gerar array de coeficientes (array_coeficientes)
    # Contém A, B, C, D tal que a função aproximadora é definida como y = A * sin(B*x + D) + C

    A, B, C, D = float(np.hypot(a, b)), freq_final, c, float(np.arctan2(b, a))

    array_coeficientes = np.array([A, B, C, D])

    # Gerar função senoidal aproximadora simbólica (func_aprox)

    x_sym = sp.Symbol("x")
    func_aprox = A*sp.sin(B*x_sym + D) + C

    if expr:
        print(f"Função Senoidal Aproximadora: y = {A:.4f} * sin({B:.4f}x + {D:.4f}) + {C:.4f}")

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
    
    # Retornar o array de coeficientes (array_coeficientes)

    return array_coeficientes

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

    # Verifica se o tamanho da lista dos valores de x e valores de y é igual

    if len(valores_x) != len(valores_y):
        raise ValueError("As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho.")
    
    # Verifica se os valores de y são positivos

    for y in valores_y:
        if y <= 0:
            raise ValueError("A lista de valores de y possui valor(es) não postivos(s).")

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

# Função de Ajuste Logaritimo

def ajuste_logaritmo(valores_x:list, valores_y:list, plt_grafico: bool = True):
    '''
    Calcula o ajuste logaritmo y = a + b * ln(x) para os dados (valores_x, valores_y) pela linearização do modelo (y = a + b*x', onde x' = ln(x)) e
    aplica o Método dos Mínimos Quadrados sobre os dados transformados por meio da função de ajuste linear.

    Além disso, a função exibe um gráfico de dispersão dos pontos e da curva de ajuste.

    Argumentos:
        valores_x (list): Lista de valores da variável independente.
        valores_y (list): Lista de valores da variável dependente.
        plt_grafico (bool, opcional): True (padrão) se o gráfico deve ser plotado, False caso contrário

    Retorna:
        tuple: (a, b), contendo o coeficiente (a) e o coeficiente logarítmico (b) da curva de ajuste.
    '''

    # Verifica se o tamanho da lista dos valores de x e valores de y é igual

    if len(valores_x) != len(valores_y):
        raise ValueError("As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho.")
    
    # Verifica se os valores de x são positivos

    for x in valores_x:
        if x <= 0:
            raise ValueError("A lista de valores de x possui valor(es) não positivos(s).")

    # Transforma o ajuste logaritmo em um ajuste linear

    ln_valores_x = [math.log(x) for x in valores_x]
    b, a = ajuste_linear(ln_valores_x, valores_y, False)

    # Plot do gráfico

    if plt_grafico:
        plt.figure()
        
        x_func = np.linspace(min(valores_x), max(valores_x), 200)
        y_func = a + b * np.log(x_func)

        plt.scatter(valores_x, valores_y, color="blue", marker="o", label="Dados Fornecidos")
        plt.plot(x_func, y_func, color="black", linewidth=2, label="Curva de Ajuste Logaritmo")

        plt.title("Gráfico do Ajuste Logaritmo")
        plt.xlabel("Eixo x")
        plt.ylabel("Eixo y")
        plt.margins(x=0.1, y=0.1)
        plt.grid(True)
        plt.legend()
        plt.show()

    return a, b

# Função de Ajuste Múltiplo

def ajuste_multiplo(valores_var: list[list[float]], valores_z: list[float], expr: bool = True) -> np.ndarray:
    """
    Realiza o ajuste múltiplo para a variável dependente valores_z a partir dos dados valores_var pelo Método dos Mínimos Quadrados (MMQ).

    A função monta e resolve o sistema (XᵀX)c = Xᵀy para c, obtendo os coeficientes da função múltipla que mais se aproxima dos dados pelo MMQ. 
    Opcionalmente, a função também exibe a forma simbólica da expressão polinomial resultante (func_aprox).

    Argumentos:
        valores_var (list): Lista de valores das variáveis independentes.
        valores_z (list): Lista de valores da variável dependente.
        expr (bool, opcional): Se True (padrão), exibe a função simbólica aproximadora; se False, não exibe.

    Retorna:
        numpy.ndarray: array_coeficientes, contendo os coeficientes da função múltipla na ordem em que as variáveis foram apresentadas em valores_var.
    """

    # Construir a matriz de valores das variáveis (x_matriz)

    xm_temp = np.array(valores_var)
    x_matriz = np.insert(xm_temp, 0, 1, axis=1)
    
    # Construir a matriz dos valores de z (z_matriz)

    z_matriz = np.array(valores_z)

    if x_matriz.shape[0] != len(z_matriz): # A quantidade de valores das variáveis independentes deve ser igual à quantidade de valores da variável dependente fornecida
        raise ValueError("Número de linhas de X e número de valores de Z não coincidem.")

    # Construir a matriz de parâmetros (array_coeficientes)

    matriz_T = x_matriz.T

    array_coeficientes = np.linalg.solve(matriz_T @ x_matriz, matriz_T @ z_matriz)
    array_coeficientes = array_coeficientes.ravel()

    # Gerar função aproximadora simbólica para regressão múltipla

    if expr:
        qtd_var = x_matriz.shape[1] - 1
        ind_fin = qtd_var + 1

        x_sym = sp.symbols(f"x1:{ind_fin}")
        func_aprox = array_coeficientes[0]

        for i in range(qtd_var):
            func_aprox += array_coeficientes[i + 1]*x_sym[i]

        print(f"Função Aproximadora para Regressão Múltipla: {func_aprox}")

    # Retornar o array de coeficientes

    return array_coeficientes

# Função de Avaliação do Ajuste

def avaliar_ajuste(valores_x: list[float], valores_y: list[float], criterio: str, modelo: str, coeficientes: tuple | np.ndarray):
    """Avalia um modelo de ajuste por meio de um ou mais critérios

    Argumentos:
        valores_x (list): Lista de valores da variável independente.
        valores_y (list): Lista de valores da variável dependente.
        criterio (str): Critério de avaliação ("R2", "R2A", "AIC", "AICc", "BIC", "all")
        modelo (str): Modelo utilizado ("linear", "polinomial", "exponencial", "logaritmo", "senoidal")
        coeficientes (tuple | np.ndarray): Coeficientes do modelo

    Retorna:
        float | tuple: Valor do critério (float) ou uma tupla com todos os critérios (caso criterio="all")
    """

    # Verifica se o tamanho da lista dos valores de x e valores de y é igual

    if len(valores_x) != len(valores_y):
        raise ValueError("As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho.")
    
    # Verifica se o critério é válido

    if criterio not in ("R2", "R2A", "AIC", "AICc", "BIC", "all"):
        raise ValueError("Critério desconhecido. Use R2, R2A , AIC, AICc, BIC, all")
    
    # Verifica se o modelo é válido

    if modelo not in ("linear", "polinomial", "exponencial", "logaritmo", "senoidal"):
        raise ValueError("Modelo desconhecido. Use linear, polinomial, exponencial, logaritmo, senoidal")

    # Transforma os valores em arrays

    valores_x = np.array(valores_x)
    valores_y = np.array(valores_y)
    n = len(valores_x)

    # Adquire os valores de y que a aproximação forneceu

    if modelo == "linear":
        qtd_coeficientes = 2
        y_modelo = coeficientes[0] * valores_x + coeficientes[1]

    elif modelo == "polinomial":
        qtd_coeficientes = len(coeficientes)
        y_modelo = np.polynomial.polynomial.polyval(valores_x, coeficientes)

    elif modelo == "exponencial":
        qtd_coeficientes = 2
        y_modelo =  coeficientes[1] * np.exp(coeficientes[0] * valores_x)

    elif modelo == "logaritmo":
        qtd_coeficientes = 2
        y_modelo = coeficientes[0] + coeficientes[1] * np.log(valores_x)

    elif modelo == "senoidal":
        qtd_coeficientes = 4
        y_modelo = coeficientes[0] * np.sin(coeficientes[1] * valores_x + coeficientes[3]) + coeficientes[2]

    # Calcula o resíduo quadrático e o resíduo total (tratando o caso igual a 0)

    media_y = np.mean(valores_y)
    RST = 1e-12 if np.sum((valores_y - media_y)**2) == 0 else np.sum((valores_y - media_y)**2)
    RSS = max(np.sum((valores_y - y_modelo)**2), 1e-12) # Evita problemas no log

    R2 = 1 - (RSS/RST)

    # Calcula os critérios

    if (n - qtd_coeficientes - 1) <= 0 and criterio in ("R2A", "AICc", "all"):
        raise ZeroDivisionError("Não é possível calcular o critério solicitado.")

    if criterio == "R2":
        return R2
    
    elif criterio == "R2A":
            return 1 - ((1- R2) * (n - 1) / (n - qtd_coeficientes - 1))
            
    elif criterio == "AIC":
        return n * np.log(RSS / n) + 2 * qtd_coeficientes
    
    elif criterio == "AICc":
        AIC = n * np.log(RSS / n) + 2 * qtd_coeficientes
        return AIC + (2* qtd_coeficientes * (qtd_coeficientes + 1)) / (n - qtd_coeficientes - 1)

    elif criterio == "BIC":
        return n * np.log(RSS / n) + qtd_coeficientes * np.log(n)
    
    elif criterio == "all":
        AIC = n * np.log(RSS / n) + 2 * qtd_coeficientes
        BIC = n * np.log(RSS / n) + qtd_coeficientes * np.log(n)
        R2A = 1 - ((1- R2) * (n - 1) / (n - qtd_coeficientes - 1))
        AICc = AIC + (2* qtd_coeficientes * (qtd_coeficientes + 1)) / (n - qtd_coeficientes - 1)

        return (R2, R2A, AIC, AICc, BIC)

# Função de Melhor Ajuste

def melhor_ajuste(valores_x: list[float], valores_y: list[float], criterio: str, exibir_todos: bool = True, plt_grafico: bool = True):
    """
    Fornece o melhor ajuste (linear ou polinomial) para a variável dependente valores_y a partir dos dados valores_x.

    A função calcula a Soma dos Quadrados Totais (SST) e encontra os ajustes linear e polinomial (grau 2 a 10)
    a partir das funções ajuste_linear e ajuste_polinomial. Além disso, calcula seus respectivos valores de 
    Soma dos Quadrados dos Resíduos (SSE).

    Após esses processos, a função calcula o R^2, R^2 ajustado, AIC, AICc e BIC, considerando que os resíduos do modelo seguem
    uma distribuição normal com variância constante.

    Por fim, ela retorna o ajuste mais apropriado quanto ao critério escolhido e o valor deste para essa aproximação.
        Nesse sentido, se o critério é R^2 ou R^2 ajustado, é retornado o ajuste cujo valor para o critério é o maior.
        Já se o critério é AIC, AICc ou BIC, é retornado o ajuste cujo valor para o critério é o menor.

    Opcionalmente, a função também exibe um gráfico de dispersão dos pontos com o polinômio/reta de ajuste junto à 
    forma simbólica da expressão polinomial resultante (func_aprox). 
    Além disso, há também a opção de retornar os valores dos outros critérios para o ajuste sugerido.

    Argumentos:
        valores_x (list): Lista de valores da variável independente.
        valores_y (list): Lista de valores da variável dependente.
        criterio (str): Critério escolhido dentre as opções "R2", "R2A" (R^2 ajustado), "AIC", "AICc" e "BIC" para sugestão do modelo.
        exibir_todos (bool, opcional): Se True (padrão), exibe os valores dos outros critérios para o modelo sugerido; se False, não exibe.
        plt_grafico (bool, opcional): Se True (padrão), exibe o gráfico de ajuste; se False, não exibe.

    Retorna:
        str: aprox_escolhida, representando o nome do modelo escolhido ("linear" ou "polinomial grau N").
        dict: funcs[aprox_escolhida], contendo as principais informações do modelo sugerido.
    """

    # Condições de Início da Função

    if len(valores_x) != len(valores_y): # A quantidade de valores da variável independente deve ser igual à quantidade de valores da variável dependente fornecida
        raise ValueError("As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho.")
    
    if criterio not in ("R2", "R2A", "AIC", "AICc", "BIC"):
        raise ValueError("Critério deve ser escolhido dentre as seguintes opções: R2, R2A, AIC, AICc, BIC")

    # Obter os parâmetros dos ajustes linear e polinomial (grau 2 a 10)

    funcs = dict()

    funcs["linear"] = {"params": np.array([par for par in ajuste_linear(valores_x, valores_y, plt_grafico=False)])}

    for grau in range(2, 11):
        funcs[f"polinomial grau {grau}"] = {"params": ajuste_polinomial(valores_x, valores_y, grau, plt_grafico=False, expr=False)}

    # Calcular SST

    valores_y = np.array(valores_y)
    valores_x = np.array(valores_x)

    media_y = np.mean(valores_y)
    copia_y = valores_y.copy()

    SST = np.sum((copia_y - media_y)**2)

    # Ajuste Linear

    # Ajuste Linear - Calcular SSE

    y_lin = np.array(funcs["linear"]["params"][0]*valores_x + funcs["linear"]["params"][1])

    SSE_lin = np.sum((copia_y - y_lin)**2)
    SSE_lin = max(SSE_lin, 1e-12) # Evita problemas no cálculo do log(SSE_lin / n) durante a obtenção dos valores dos critérios

    funcs["linear"]["SSE"] = SSE_lin

    # Ajustes Polinomiais

    for i in range(2, 11):

        # Ajustes Polinomiais - Calcular SSE

        y_pol = np.polynomial.polynomial.polyval(valores_x, funcs[f"polinomial grau {i}"]["params"])

        SSE_pol = np.sum((copia_y - y_pol)**2)
        SSE_pol = max(SSE_pol, 1e-12) # Evita problemas no cálculo do log(SSE_pol / n) durante a obtenção dos valores dos critérios

        funcs[f"polinomial grau {i}"]["SSE"] = SSE_pol

    # Calcular R^2, R^2 ajustado, AIC, AICc e BIC para Ajustes Linear e Polinomiais

    n = len(valores_x)

    lista_ajustes = ["linear"] + [f"polinomial grau {i}" for i in range(2, 11)]

    for ajuste in lista_ajustes:

        if ajuste == "linear":
            p = 2

        else:
            grau = int(ajuste.split()[-1])     
            p = grau + 1
        
        # Calcular R^2

        SSE = funcs[ajuste]["SSE"]

        R2 = 1 - (SSE/SST)
        funcs[ajuste]["R2"] = R2

        # Calcular R^2 Ajustado

        R2A = 1 - ((1 - R2) * (n - 1)/(n - p - 1))
        funcs[ajuste]["R2A"] = R2A

        # Calcular AIC

        AIC = n * np.log(SSE / n) + 2 * p
        funcs[ajuste]["AIC"] = AIC

        # Calcular AICc

        if n - p - 1 > 0:
            AICc = AIC + (2*p*(p+1))/(n - p - 1)
        else:
            AICc = np.inf

        funcs[ajuste]["AICc"] = AICc

        # Calcular BIC

        BIC = n * np.log(SSE / n) + p*np.log(n)
        funcs[ajuste]["BIC"] = BIC
    
    # Encontrar a aproximação mais adequada com base no critério escolhido
    
    if criterio == "R2" or criterio == "R2A":
        funcs_ordenadas = dict(sorted(funcs.items(), key=lambda item: item[1][criterio], reverse=True))
    elif criterio == "AIC" or criterio == "AICc" or criterio == "BIC":
        funcs_ordenadas = dict(sorted(funcs.items(), key=lambda item: item[1][criterio]))
    
    aprox_escolhida = next(iter(funcs_ordenadas))
    
    print(f"Modelo sugerido: Aproximação {aprox_escolhida}")
    print(f"{criterio}: {funcs[aprox_escolhida][criterio]:.6f}\n")

    # Exibir todos o valores de todos os critérios da aproximação escolhida

    if exibir_todos:
        lista_criterios = [c for c in ["R2", "R2A", "AIC", "AICc", "BIC"] if c != criterio]
        for crit in lista_criterios:
            if crit == "R2A":
                print(f"R2 Ajustado: {funcs[aprox_escolhida][crit]}")
            else:
                print(f"{crit}: {funcs[aprox_escolhida][crit]}")

    # Plotar o gráfico e exibir a expressão simbólica da aproximação escolhida

    if plt_grafico:
        if aprox_escolhida == "linear":
            ajuste_linear(valores_x, valores_y, plt_grafico=True)
        else:
            grau = int(aprox_escolhida.split()[-1])
            ajuste_polinomial(valores_x, valores_y, grau, plt_grafico=True, expr=False)
    
    # Retornar a aproximação escolhida e seu dicionário de informações
    
    return aprox_escolhida, funcs[aprox_escolhida]