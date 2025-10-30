import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

"""
A função abaixo calcula o Polinômio Interpolador de Lagrange para um
conjunto de pontos (x_i, f(x_i)) dados, retornando a expressão polinomial
simplificada e simbólica usando a biblioteca SymPy.
"""
def interpolacao_polinomial(tupla_de_pontos, plotar = False) -> sp.Expr:
    
    # Aplicamos um tste de entrada, mais precisamente verificamos se a lista de entrada está vazia. Se estiver, retorna uma mensagem de erro.
    if not tupla_de_pontos:
        return "Lista de pontos vazia"
    

    # Quebramos a lista de tuplas [(x_0, f(x_0)), (x_1, f(x_1)), ...] em dois arrays separados.
    # O dtype=float converte os elementos do array para float.
    X = np.array([p[0] for p in tupla_de_pontos], dtype=float)
    Y = np.array([p[1] for p in tupla_de_pontos], dtype=float)
    n = len(X)

    # Definimos aqui 'x' como uma variável simbólica usando SymPy
    # para que consigamos manipular os resultados obtidos na cons-
    # trução do polinômio de forma algébrica.
    x = sp.Symbol('x')


    # Inicializamos o Polinômio Interpolador de Lagrange P(x)
    P_x = 0
    
    # Inicializamos a lista L (Polinômios Construtores Lagrangianos L_k(x)),
    # com n 1's, onde n é o grau máximo posível para esse polinômio.
    # Ademais, inicializamos com n 1's pois L acumulará um produto de termos.
    L = [1]*n
    
   
    # O loop externo (k) itera sobre cada ponto (x_k, f(x_k)) para construir L_k(x),isto é,
    # gerará os n Polinômios Construtores Lagrangianos que apendaremos em L.
    for k in range(n):
        # O loop interno (i) constrói o k-ésimo Polinômio Construtor L_k(x) via produtório,isto é,
        # L_k(x) = Prod_{i != k} [ (x - x_i) / (x_k - x_i) ]
        for i in range(n):
            # Aqui aplicamos a condição i != k devida a fórmula de Lagrange,
            # visando evitar a divisão por isso, ou seja, evitamos que o 
            # denominador (X[k] - X[i]) se anule.
            if i != k:
                # Multiplimos cumulativamente: L[k] = L[k] * proximo_termo
                # O proximo_termo é (x - X_i) / (X_k - X_i).
                L[k] *= (x - X[i])/(X[k] - X[i])
                
    
    # O loop final (j) soma os termos para obter o Polinômio Interpolador de Lagrange P(x):
    # P(x) = Sum_{j=0}^{n-1} [ Y_j * L_j(x) ]
    for j in range(n):
        P_x += Y[j] * L[j]
    
    polinomio_simplificado = sp.simplify(P_x)


    # ====================================================================================
    # ====================================================================================



    if plotar:
        # Definimos uma função interna que plotará os pontos e o Polinômio Interpolador
        def plotar_interpolacao(pontos, polinomio_simplificado):
            """
            Esta função gera o gráfico do polinômio interpolador e dos pontos originais.
            
            Argumentos:
                pontos (lista de tupla): Lista de pontos originais (x_i, f(x_i)).
                polinomio_simplificado (sympy.Expr): O polinômio P(x) retornado pela função.
            """
            
            # 1. Preparação dos Dados Originais (para plotagem dos marcadores)
            X_pontos = np.array([p[0] for p in pontos])
            Y_pontos = np.array([p[1] for p in pontos])
            
            # 2. Convertemos a expressão simbólica para uma função numérica
            # sp.lambdify converte a expressão SymPy (em 'x') para uma função NumPy rápida.
            x_simbolico = sp.Symbol('x')
            P_x_numerico = sp.lambdify(x_simbolico, polinomio_simplificado, 'numpy')
            
            # 3. Geração do Espaço Amostral para a Curva
            # Define o intervalo de plotagem ligeiramente maior que os pontos dados.
            x_min = np.min(X_pontos) - 0.5
            x_max = np.max(X_pontos) + 0.5
            
            # Gera 1000 pontos uniformemente espaçados para a curva
            X_curva = np.linspace(x_min, x_max, 1000)
            
            # 4. Avaliação do Polinômio
            Y_curva = P_x_numerico(X_curva)
            
            # 5. Plotagem com Matplotlib
            plt.figure(figsize=(10, 6))
            
            # Plota a curva do Polinômio (linha contínua)
            plt.plot(X_curva, Y_curva, label=f'$P(x) = {polinomio_simplificado}$', color='blue')
            
            # Plota os Pontos Originais (marcadores vermelhos)
            plt.scatter(X_pontos, Y_pontos, label='Pontos de Interpolação', color='red', marker='o', zorder=5)
            
            # Adicionando rótulos e título
            plt.title('Interpolação Polinomial de Lagrange')
            plt.xlabel('Eixo X')
            plt.ylabel('Eixo Y')
            plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.show()

        plotar_interpolacao(tupla_de_pontos, polinomio_simplificado)

        # Retorna a expressão final simplificada, expandindo e combinando os termos 
        # para a forma padrão de um polinômio (e.g., a_n*x^n + a_{n-1}*x^{n-1} + . . . + a_1*x + a_0).
        return polinomio_simplificado
    else:
        # Retorna a expressão final simplificada, expandindo e combinando os termos 
        # para a forma padrão de um polinômio (e.g., a_n*x^n + a_{n-1}*x^{n-1} + . . . + a_1*x + a_0).
        return polinomio_simplificado













            