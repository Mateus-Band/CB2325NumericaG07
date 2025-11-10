import math
from functools import reduce

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# Função Auxiliar para Plotar Gráficos

def plotar_grafico(
        valores_x: list[float], 
        valores_y: list[float], 
        func_sym: sp.Expr, 
        titulo: str, 
        qtd_pontos: int = 200
):
    """
    Exibe um gráfico de dispersão dos pontos com a função de ajuste

    Argumentos:
        valores_x (list): Lista de valores da variável independente.
        valores_y (list): Lista de valores da variável dependente.
        func_sym (sp.Expr): Expressão simbólica da função aproximadora.
        titulo (str): Título do gráfico a ser gerado.
        qtd_pontos(int): Quantidade de pontos a ser inserida no 
        gráfico da função.
    """

    # Tratar a função simbólica

    x_sym = sp.Symbol("x")
    f = sp.lambdify(x_sym, func_sym, "numpy")

    # Gerar os pontos

    x_func = np.linspace(min(valores_x), max(valores_x), qtd_pontos)
    y_func = np.array(f(x_func))
    if y_func.shape == ():
            y_func = np.full_like(x_func, y_func)

    # Plotar os gráficos

    plt.scatter(valores_x, valores_y, color="blue", marker="o", 
        label="Dados Fornecidos")

    plt.plot(x_func, y_func, color="black", linewidth=2, 
        label="Função Aproximadora")

    plt.title(titulo)
    plt.xlabel("Eixo x")
    plt.ylabel("Eixo y")
    plt.margins(x=0.1, y=0.1)
    plt.grid(True)
    plt.legend()
    plt.show()


# Função de Ajuste Linear

def ajuste_linear(
        valores_x: list, 
        valores_y: list, 
        plt_grafico: bool = True, 
        expr: bool = False
) -> tuple:
    """
    Calcule o ajuste linear y = ax + b pelo Método dos Mínimos Quadrados (MMQ).

    Além disso, a função exibe um gráfico de dispersão dos pontos e da 
    reta de ajuste.

    Argumentos:
        valores_x (list): Lista de valores da variável independente.
        valores_y (list): Lista de valores da variável dependente.
        plt_grafico (bool, opcional): True (padrão) se o gráfico deve 
            ser plotado, False caso contrário.
        expr (bool, opcional): True se a expressão deve ser mostrada,
            False (padrão), caso contrário

    Retorna:
        tuple: (a, b), contendo o coeficiente angular (a) e o 
            coeficiente linear (b) da reta de ajuste.

    Raises:
        ValueError: Se a lista de valores x e y tiverem tamanhos 
            diferentes ou a variância dos valores de x for 0.
    """

    if len(valores_x) != len(valores_y):
        raise ValueError("As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho.")

    x_medio = reduce(lambda x, y: x + y, valores_x) / len(valores_x)
    y_medio = reduce(lambda x, y: x + y, valores_y) / len(valores_y)

    # Cálculo da covariância de x e y e da variância

    cov_xy = 0
    var_x = 0

    for i in range(len(valores_x)):
        cov_xy += (valores_x[i] - x_medio) * (valores_y[i] - y_medio)
        var_x += (valores_x[i] - x_medio) ** 2

    if var_x == 0:
        raise ValueError("A variância de valores_x é zero. Não é possível calcular o ajuste.")

    # Cálculo do coeficientes

    a = cov_xy / var_x           
    b = y_medio - a * x_medio 

    # Print da expressão

    if expr:
        print(f"Função linear aproximadora: y = {a:.4f}x + {b:.4f}") 

    # Plot do gráfico

    if plt_grafico:
        x_sym = sp.Symbol("x")
        y_func = a * x_sym + b

        plotar_grafico(
            valores_x,
            valores_y,
            y_func,
            "Gráfico do Ajuste Linear"
        )

    return a, b


# Função de Ajuste Polinomial

def ajuste_polinomial(
        valores_x: list[float] | np.ndarray, 
        valores_y: list[float] | np.ndarray, 
        grau_pol: int, 
        plt_grafico: bool = True, 
        expr: bool = False
) -> np.ndarray:
    """
    Realiza o ajuste polinomial pelo Método dos Mínimos Quadrados (MMQ).

    A função obtém os coeficientes do polinômio de grau especificado 
    pelo usuário que mais se aproxima dos dados pelo MMQ. 

    Opcionalmente, a função também exibe:
        - Um gráfico de dispersão dos pontos com o polinômio de ajuste;
        - A forma simbólica do polinômio de ajuste (func_aprox).

    Argumentos:
        valores_x (list | np.ndarray): 
            Valores da variável independente.
        valores_y (list | np.ndarray):
            Valores da variável dependente.
        grau_pol (int): 
            Grau do polinômio ao qual os dados serão ajustados.
        plt_grafico (bool, opcional): 
            Se True (padrão), exibe o gráfico de ajuste; 
            se False, não exibe.
        expr (bool, opcional): 
            Se True, exibe a função simbólica aproximadora; 
            se False (padrão), não exibe.

    Retorna:
        numpy.ndarray: 
            array_coeficientes, contendo os coeficientes do polinômio 
            em ordem crescente do grau da variável associada.
    """

    # Condições de início

    if len(valores_x) != len(valores_y):
        raise ValueError(
            "As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho."
        )
    
    if grau_pol < 0:
        raise ValueError("grau_pol deve ser não negativo.")

    # Construir matriz de Vandermonde (x_matriz)

    x_matriz = np.array(
        [[valor ** i for i in range(grau_pol + 1)] for valor in valores_x]
    )
    
    # Construir a matriz dos valores de y (y_matriz)

    y_matriz = np.array(valores_y)

    # Obter os coeficientes (array_coeficientes)

    array_coeficientes, *_ = np.linalg.lstsq(x_matriz, y_matriz, rcond=None)

    # Gerar função polinomial aproximadora simbólica (func_aprox)

    x_sym = sp.Symbol("x")
    func_aprox = 0

    for i in range(len(array_coeficientes)):
        func_aprox += array_coeficientes[i] * x_sym ** i

    if expr:
        print(f"Função Polinomial Aproximadora: {func_aprox}")

    # Plotar o gráfico

    if plt_grafico:
        plotar_grafico(
            valores_x, 
            valores_y, 
            func_aprox, 
            f"Gráfico dos Dados Fornecidos e da Função "
            f"Polinomial Aproximadora de Grau {grau_pol}"
        )
    
    # Retornar os coeficientes (array_coeficientes)

    return array_coeficientes


# Função de Ajuste Senoidal

def ajuste_senoidal(
        valores_x: list[float] | np.ndarray, 
        valores_y: list[float] | np.ndarray, 
        T_aprox: float | None = None,
        plt_grafico: bool = True, 
        expr: bool = False
) -> np.ndarray:
    """
    Realiza o ajuste senoidal pelo Método dos Mínimos Quadrados (MMQ).

    Modelo adotado:
        y = A * sin(B * x + C) + D. 

    Forma linearizada:
        y = a * sin(B*x) + b * cos(B*x) + d
        onde a = A * cos(C), b = A * sin(C), d = D.

    A função estima a frequência B inicialmente 
    pela aproximação do período pelo usuário. Este é inserido opcionalmente
    como parâmetro da função ou, no caso em que não o é, é captado
    via input. Nesse caso, será exibido o gráfico com os dados fornecidos, 
    em que o usuário deve indicar o período aproximado visualmente. 
    Em seguida, ela testa diversas frequências 
    em torno da frequência inicial pelo MMQ

    Por fim, ao encontrar a frequência que resulta no menor erro quadrático, 
    ela gera a lista de coeficientes A, B, C e D 
    da função senoidal aproximadora.

    Opcionalmente, a função também exibe:
        - Um gráfico de dispersão dos pontos com a senóide de ajuste;
        - A forma simbólica da senóide de ajuste (func_aprox).

    Argumentos:
        valores_x (list | np.ndarray): 
            Valores da variável independente.
        valores_y (list | np.ndarray): 
            Valores da variável dependente.
        T_aprox (float | None, opcional): Período aproximado da senóide. 
            Se for fornecido (float), será usado diretamente.  
            Se for None (padrão), o período será solicitado ao usuário via input().
        plt_grafico (bool, opcional): 
            Se True (padrão), exibe o gráfico de ajuste; 
            se False, não exibe.
        expr (bool, opcional): 
            Se True, exibe a função senoidal simbólica aproximadora; 
            se False (padrão), não exibe.

    Retorna:
        numpy.ndarray: 
            array_coeficientes, contendo os coeficientes A, B, C e D.
    """
    # Condição de início

    if len(valores_x) != len(valores_y):
        raise ValueError("As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho.")

     # Plotar o gráfico de dispersão dos dados fornecidos 
     # para que o usuário indique o período aproximado percebido na amostra

    plt.scatter(valores_x, valores_y, color="blue", marker="o", label="Dados Fornecidos")
    plt.title("Gráfico para Aproximação do Período")
    plt.xlabel("Eixo x")
    plt.ylabel("Eixo y")
    plt.margins(x=0.1, y=0.1)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Captar o período aproximado e cálculo da frequência aproximada 

    if T_aprox is None:
        T_aprox = float(input("Digite o Período Aproximado: "))
    freq_aprox = (2*np.pi) / T_aprox

    # Identificar a frequência de menor erro quadrático 
    # dentro de um intervalo próximo à frequência aproximada.
    # São testadas 400 frequências igualmente espaçadas em torno da frequência inicial.

    freq_list = np.linspace(freq_aprox * 0.5, freq_aprox * 1.5, 400)
    erros = dict()

    for freq in freq_list:

        # Construir a matriz X das funções base (x_matriz) 
        # no formato [sin(Bx), cos(Bx), 1]

        x_matriz = np.array([
            [np.sin(freq * valor), np.cos(freq * valor), 1] 
            for valor in valores_x])

        matriz_sin = np.array([np.sin(freq * valor) for valor in valores_x])
        matriz_cos = np.array([np.cos(freq * valor) for valor in valores_x])
        
        # Construir a matriz dos valores de y (y_matriz)

        y_matriz = np.array(valores_y)

        # Construir a matriz de parâmetros iniciais (coeff_iniciais), ou seja: a, b e d

        try:
            coeff_iniciais, *_ = np.linalg.lstsq(x_matriz, y_matriz, rcond=None)
        except np.linalg.LinAlgError:
            continue  # pula frequências com sistema singular

        a, b, d = coeff_iniciais[0], coeff_iniciais[1], coeff_iniciais[2]

        # Calcular o erro para a frequência em questão

        erro = np.linalg.norm(
            y_matriz - (a * matriz_sin + b * matriz_cos + d))**2

        erros[freq] = (erro, (a, b, d))
    
    # Obter a frequência de menor erro (freq_final) e seus parâmetros (a, b e c)

    erros_ordenados = dict(sorted(erros.items(), key=lambda item: item[1][0]))
    freq_final = next(iter(erros_ordenados))

    a = erros_ordenados[freq_final][1][0]
    b = erros_ordenados[freq_final][1][1]
    d = erros_ordenados[freq_final][1][2]

    # Gerar array de coeficientes (array_coeficientes)
    # Contém A, B, C, D tal que a função aproximadora é definida como 
    # y = A * sin(B*x + C) + D.

    A = float(np.hypot(a, b))
    B = freq_final
    C = float(np.arctan2(b, a))
    D = d

    array_coeficientes = np.array([A, B, C, D])

    # Gerar função senoidal aproximadora simbólica (func_aprox)

    x_sym = sp.Symbol("x")
    func_aprox = A * sp.sin(B * x_sym + C) + D

    if expr:
        print(f"Função Senoidal Aproximadora: "
              f"y = {A:.4f} * sin({B:.4f}x + {C:.4f}) + {D:.4f}"
        )

    # Plotar o Gráfico

    if plt_grafico:
        plotar_grafico(
            valores_x, 
            valores_y, 
            func_aprox, 
            "Gráfico dos Dados Fornecidos e da Função Senoidal Aproximadora",
            qtd_pontos=600
        )
    
    # Retornar o array de coeficientes (array_coeficientes)

    return array_coeficientes


# Função de Ajuste Exponencial

def ajuste_exponencial(
        valores_x: list, 
        valores_y: list, 
        plt_grafico: bool = True, 
        expr: bool = False
) -> tuple:
    """
    Calcule o ajuste exponencial y = b * e^(a*x)
     
    Isso é feito pela linearização do modelo ln(y) = ln(b) + a*x e 
    aplicando o Método dos Mínimos Quadrados sobre os dados 
    transformados por meio da função de ajuste linear. Além disso, a 
    função exibe um gráfico de dispersão dos pontos e da curva de 
    ajuste.

    Argumentos:
        valores_x (list): Lista de valores da variável independente.
        valores_y (list): Lista de valores da variável dependente.
        plt_grafico (bool, opcional): True (padrão) se o gráfico deve 
            ser plotado, False caso contrário
        expr (bool, opcional): True se a expressão deve ser mostrada,
            False (padrão), caso contrário
    Retorna:
        tuple: (a, b), contendo o coeficiente do expoente (a) e o 
            coeficiente multiplicativo (b) da curva de ajuste.
    
    Raises:
        ValueError: Se a lista de valores x e y tiverem tamanhos 
            diferentes ou a lista de valores y tiver valores não positivos.
    """

    if len(valores_x) != len(valores_y):
        raise ValueError("As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho.")

    for y in valores_y:
        if y <= 0:
            raise ValueError("A lista de valores de y possui valor(es) não postivos(s).")

    # Transforma o ajuste exponencial em um ajuste linear

    ln_valores_y = [math.log(y) for y in valores_y]
    a, b_aux = ajuste_linear(valores_x, ln_valores_y, False)
    b = math.exp(b_aux)

    # Print da expressão

    if expr:
        print(f"Função exponencial aproximadora: y = {b:.4f} * e^({a:.4f}x)")

    # Plot do gráfico

    if plt_grafico:
        x_sym = sp.Symbol("x")
        y_func = b * sp.exp(a * x_sym)

        plotar_grafico(
            valores_x,
            valores_y,
            y_func,
            "Gráfico do Ajuste Exponencial"
        )

        '''
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
        plt.show()'''

    return a, b


# Função de Ajuste Logaritimo

def ajuste_logaritmo(
        valores_x: list, 
        valores_y: list, 
        plt_grafico: bool = True, 
        expr: bool = False
) -> tuple:
    '''
    Calcula o ajuste logaritmo y = a + b * ln(x) 
    
    Isso é feito pela linearização do modelo 
    (y = a + b*x', onde x' = ln(x)) e aplica o Método dos Mínimos 
    Quadrados sobre os dados transformados por meio da função de 
    ajuste linear. Além disso, a função exibe um gráfico de dispersão
    dos pontos e da curva de ajuste.

    Argumentos:
        valores_x (list): Lista de valores da variável independente.
        valores_y (list): Lista de valores da variável dependente.
        plt_grafico (bool, opcional): True (padrão) se o gráfico deve
            ser plotado, False caso contrário.
        expr (bool, opcional): True se a expressão deve ser mostrada,
            False (padrão), caso contrário

    Retorna:
        tuple: (a, b), contendo o coeficiente (a) e o coeficiente 
            logarítmico (b) da curva de ajuste.

    Raises:
        ValueError: Se a lista de valores x e y tiverem tamanhos 
            diferentes ou a lista de valores x tiver valores não positivos.
    '''

    if len(valores_x) != len(valores_y):
        raise ValueError("As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho.")

    for x in valores_x:
        if x <= 0:
            raise ValueError("A lista de valores de x possui valor(es) não positivos(s).")

    # Transforma o ajuste logaritmo em um ajuste linear

    ln_valores_x = [math.log(x) for x in valores_x]
    b, a = ajuste_linear(ln_valores_x, valores_y, False)

    # Print da expressão

    if expr:
        print(f"Função logaritmo aproximadora: y = {a:.4f} + {b:.4f} * ln(x)")

    # Plot do gráfico

    if plt_grafico:
        x_sym = sp.Symbol("x")
        y_func = a + b * sp.ln(x_sym)

        plotar_grafico(
            valores_x,
            valores_y,
            y_func,
            "Gráfico do Ajuste Logaritmo"
        )

    return a, b


# Função de Ajuste Múltiplo

def ajuste_multiplo(
        valores_var: list[list[float]] | np.ndarray, 
        valores_z: list[float] | np.ndarray,
        incluir_intercepto : bool = True, 
        expr: bool = False
) -> np.ndarray:
    """
    Realiza o ajuste múltiplo pelo Método dos Mínimos Quadrados (MMQ).

    A função obtém os coeficientes da função múltipla 
    que mais se aproxima dos dados pelo MMQ. 
    Opcionalmente, a função também inclui a busca por um intercepto
    (termo independente) e exibe a forma simbólica 
    da expressão polinomial resultante (func_aprox).

    Observação: O método pressupõe que as variáveis em 'valores_var'
    não sejam fortemente relacionadas entre si e que não há
    colinearidade. Caso contrário, pode haver instabilidade 
    na regressão e/ou resultados incorretos.

    Argumentos:
        valores_var (list | np.ndarray): 
            Valores das variáveis independentes.
        valores_z (list | np.ndarray): 
            Valores da variável dependente.
        incluir_intercepto (bool, opcional): 
            Se True (padrão), inclui o termo independente na 
            solução do sistema linear; se False, não inclui.
        expr (bool, opcional): 
            Se True, exibe a função simbólica aproximadora; 
            se False (padrão), não exibe.

    Retorna:
        numpy.ndarray: array_coeficientes, contendo os coeficientes 
        da função múltipla.
            - Se incluir_intercepto = True, retorna o termo
            independente na primeira posição e os coeficientes
            relacionados às variáveis na ordem em que foram
            apresentadas em 'valores_var'.
            - Se incluir_intercepto = False, retorna apenas os
            coeficientes relacionados às variáveis na ordem em
            que foram apresentadas em 'valores_var'.
    """

    # Construir a matriz de valores das variáveis (x_matriz)

    x_matriz = np.array(valores_var, dtype=float).T

    # Construir a matriz dos valores de z (z_matriz)

    z_matriz = np.array(valores_z, dtype=float).reshape(-1, 1)

    # Condição de início

    if x_matriz.shape[0] != z_matriz.shape[0]:
        raise ValueError(
            "Número de linhas de X e número de valores de Z não coincidem."
        )
    
    # Tratar o caso com intercepto

    if incluir_intercepto:
        x_matriz = np.column_stack([np.ones(len(valores_z)), x_matriz])

    # Verificação de colinearidade

    if np.linalg.matrix_rank(x_matriz) < x_matriz.shape[1]:
        raise ValueError(
            "Colinearidade detectada por matriz mal-condicionada. "
            "Verifique os dados de entrada."
        )

    # Construir a matriz de parâmetros (array_coeficientes)

    array_coeficientes, *_ = np.linalg.lstsq(x_matriz, z_matriz, rcond=None)
    array_coeficientes = array_coeficientes.ravel()

    # Gerar função aproximadora simbólica para regressão múltipla

    if expr:
        if incluir_intercepto:
            qtd_var = x_matriz.shape[1] - 1
            ind_fin = qtd_var + 1
        
            x_sym = sp.symbols(f"x1:{ind_fin}")
            func_aprox = array_coeficientes[0]

            for i in range(qtd_var):
                func_aprox += array_coeficientes[i + 1] * x_sym[i]

        else:
            qtd_var = x_matriz.shape[1]
            ind_fin = qtd_var + 1
        
            x_sym = sp.symbols(f"x1:{ind_fin}")
            func_aprox = 0

            for i in range(qtd_var):
                func_aprox += array_coeficientes[i] * x_sym[i]
        
        print(f"Função Aproximadora para Regressão Múltipla: {func_aprox}")

    # Retornar o array de coeficientes

    return array_coeficientes


# Função de Avaliação do Ajuste

def avaliar_ajuste(
        valores_x: list[float], 
        valores_y: list[float], 
        criterio: str, 
        modelo: str, 
        coeficientes: tuple | np.ndarray
    ) -> float | tuple:
    """
    Avalie um modelo de ajuste por meio de um ou mais critérios

    Argumentos:
        valores_x (list): Lista de valores da variável independente.
        valores_y (list): Lista de valores da variável dependente.
        criterio (str): Critério de avaliação ("R2", "R2A", "AIC", 
            "AICc", "BIC", "all").
        modelo (str): Modelo utilizado ("linear", "polinomial", 
            "exponencial", "logaritmo", "senoidal").
        coeficientes (tuple | np.ndarray): Coeficientes do modelo.

    Retorna:
        float | tuple: Valor do critério (float) ou uma tupla com 
            todos os critérios (caso criterio="all").
    
    Raises:
        ValueError: Se as listas de valores x e y tiverem tamanho
            diferente ou os o criterio ou modelo forem desconhecidos.
        ZeroDivisionError: Se for impossível calcular R2A ou AICc.
    """


    if len(valores_x) != len(valores_y):
        raise ValueError("As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho.")

    if criterio not in ("R2", "R2A", "AIC", "AICc", "BIC", "all"):
        raise ValueError("Critério desconhecido. Use R2, R2A , AIC, AICc, BIC, all")

    if modelo not in ("linear", "polinomial", "exponencial", "logaritmo", "senoidal"):
        raise ValueError("Modelo desconhecido. Use linear, polinomial, exponencial, logaritmo, senoidal")

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
        y_modelo = coeficientes[1] * np.exp(coeficientes[0] * valores_x)

    elif modelo == "logaritmo":
        qtd_coeficientes = 2
        y_modelo = coeficientes[0] + coeficientes[1] * np.log(valores_x)

    elif modelo == "senoidal":
        qtd_coeficientes = 4
        y_modelo = coeficientes[0] * np.sin(coeficientes[1] * valores_x + coeficientes[3]) + coeficientes[2]

    # Calcula o resíduo quadrático e o resíduo total (tratando o caso igual a 0)

    media_y = np.mean(valores_y)
    RST = 1e-12 if np.sum((valores_y - media_y) ** 2) == 0 else np.sum((valores_y - media_y) ** 2)
    RSS = max(np.sum((valores_y - y_modelo) ** 2), 1e-12) # Evita problemas no log

    R2 = 1 - (RSS / RST)

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
        return AIC + (2 * qtd_coeficientes * (qtd_coeficientes + 1)) / (n - qtd_coeficientes - 1)

    elif criterio == "BIC":
        return n * np.log(RSS / n) + qtd_coeficientes * np.log(n)
    
    elif criterio == "all":
        AIC = n * np.log(RSS / n) + 2 * qtd_coeficientes
        BIC = n * np.log(RSS / n) + qtd_coeficientes * np.log(n)
        R2A = 1 - ((1 - R2) * (n - 1) / (n - qtd_coeficientes - 1))
        AICc = AIC + (2 * qtd_coeficientes * (qtd_coeficientes + 1)) / (n - qtd_coeficientes - 1)

        return (R2, R2A, AIC, AICc, BIC)


# Função de Melhor Ajuste

def melhor_ajuste(
        valores_x: list[float] | np.ndarray, 
        valores_y: list[float] | np.ndarray, 
        criterio: str, 
        exibir_todos: bool = False, 
        plt_grafico: bool = True, 
        expr: bool = False
):
    """
    Fornece o melhor ajuste (linear ou polinomial) para a variável valores_y.

    A função encontra os ajustes linear e polinomial (grau 2 a 10) 
    a partir das funções ajuste_linear e ajuste_polinomial. 
    Além disso, calcula seus respectivos valores 
    de Soma dos Quadrados dos Resíduos (SSE).

    Após esses processos, a função calcula o R^2, R^2 ajustado, 
    AIC, AICc e BIC através da função avaliar_ajuste,
    considerando que os resíduos do modelo seguem uma distribuição normal 
    com variância constante.

    Por fim, ela retorna o ajuste mais apropriado quanto ao critério escolhido 
    e o valor deste para essa aproximação.
    Nesse sentido, se o critério é R^2 ou R^2 ajustado, 
    é retornado o ajuste cujo valor para o critério é o maior.
    Já se o critério é AIC, AICc ou BIC, 
    é retornado o ajuste cujo valor para o critério é o menor.

    Opcionalmente, a função também exibe:
        - Um gráfico de dispersão dos pontos com a função de ajuste;
        - A forma simbólica da reta/polinômio de ajuste (func_aprox);
        - Os valores dos outros critérios para o ajuste sugerido.

    Argumentos:
        valores_x (list | np.ndarray): 
            Valores da variável independente.
        valores_y (list | np.ndarray): 
            Valores da variável dependente.
        criterio (str): 
            Critério escolhido dentre as opções: "R2", "R2A" (R^2 ajustado), 
            "AIC", "AICc" e "BIC" para sugestão do modelo.
        exibir_todos (bool, opcional): 
            Se True, exibe os valores dos outros critérios; 
            se False (padrão), não exibe.
        plt_grafico (bool, opcional): 
            Se True (padrão), exibe o gráfico de ajuste; 
            se False, não exibe.
        expr (bool, opcional): 
            Se True, exibe a função simbólica aproximadora sugerida; 
            se False (padrão), não exibe.

    Retorna:
        str: 
            aprox_escolhida, representando o nome do modelo escolhido.
        dict: 
            funcs[aprox_escolhida], contendo as principais informações 
            do modelo sugerido.
    """

    # Condições de Início da Função

    if len(valores_x) != len(valores_y):
        raise ValueError("As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho.")
    
    if criterio not in ("R2", "R2A", "AIC", "AICc", "BIC"):
        raise ValueError(
            "Critério deve ser escolhido dentre as seguintes opções: "
            "R2, R2A, AIC, AICc, BIC"
        )

    # Obter os parâmetros dos ajustes linear e polinomial (grau 2 a 10)

    funcs = dict()

    funcs["linear"] = {"params": np.array(
        [par for par in ajuste_linear(valores_x, valores_y, plt_grafico=False)]
    )}

    for grau in range(2, 11):
        funcs[f"polinomial grau {grau}"] = {"params": ajuste_polinomial(
            valores_x, valores_y, grau, plt_grafico=False, expr=False
        )}

    # Encontrar os valores dos critérios

    valores_y = np.array(valores_y)
    valores_x = np.array(valores_x)

    # Ajuste Linear

    # Ajuste Linear - Calcular SSE

    y_lin = np.array(funcs["linear"]["params"][0] * valores_x + funcs["linear"]["params"][1])

    SSE_lin = np.sum((valores_y - y_lin)**2)
    SSE_lin = max(SSE_lin, 1e-12) # Evita problemas no cálculo do log(SSE_lin / n).

    funcs["linear"]["SSE"] = SSE_lin

    # Ajustes Polinomiais

    for i in range(2, 11):

        # Ajustes Polinomiais - Calcular SSE

        y_pol = np.polynomial.polynomial.polyval(valores_x, funcs[f"polinomial grau {i}"]["params"])

        SSE_pol = np.sum((valores_y - y_pol) ** 2)
        SSE_pol = max(SSE_pol, 1e-12) # Evita problemas no cálculo do log(SSE_pol / n).

        funcs[f"polinomial grau {i}"]["SSE"] = SSE_pol

    # Obter R^2, R^2 ajustado, AIC, AICc e BIC para Ajustes Linear e Polinomiais

    lista_ajustes = ["linear"] + [f"polinomial grau {i}" for i in range(2, 11)]

    for ajuste in lista_ajustes:

        mod = "linear" if ajuste == "linear" else "polinomial"
        
        R2, R2A, AIC, AICc, BIC = avaliar_ajuste(valores_x, valores_y, "all", mod, funcs[ajuste]["params"])
        
        funcs[ajuste]["R2"] = R2
        funcs[ajuste]["R2A"] = R2A
        funcs[ajuste]["AIC"] = AIC
        funcs[ajuste]["AICc"] = AICc
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

    graf = True if plt_grafico else False
    ff = True if expr else False

    if aprox_escolhida == "linear":
        ajuste_linear(valores_x, valores_y, plt_grafico=graf)
    else:
        grau = int(aprox_escolhida.split()[-1])
        ajuste_polinomial(valores_x, valores_y, grau, plt_grafico=graf, expr=ff)   
    
    # Retornar a aproximação escolhida e seu dicionário de informações
    
    return aprox_escolhida, funcs[aprox_escolhida]