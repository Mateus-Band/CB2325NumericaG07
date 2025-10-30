import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def diff_numerica(lista:zip) -> list:
    '''
    Recebe uma sequencia de pontos da forma: [(0, 0)] (pode se fazer zip() de duas arrays ou listas com os valores x e y dos pontos),
    e retorna a derivada numerica em cada ponto,usando diferença central nos pontos centrais,
    e diferença progressiva/regressiva nas pontas.
    '''
    
    lista = list(lista)

    diff_list = lista.copy()

    for index,num in enumerate(lista):
        x,y = num
        if index in [0,len(lista)-1]: #se for um dos pontos das pontas
            if index == 0:
                next_x , next_y = lista[index + 1]
                diff_list[index] = (next_y - y)/(next_x - x)  
            
            else:
                prev_x,prev_y = lista[index - 1]
                diff_list[index] = (y - prev_y)/(x - prev_x)

        else:
            next_x , next_y = lista[index + 1]
            prev_x,prev_y = lista[index - 1]
            diff_list[index] = (next_y - prev_y)/(next_x - prev_x)
    
    return diff_list


def function_definer(lista_x,lista_y,exception=None):
    '''
    Essa função recebe duas listas e as vincula, retornando uma função vinculo que ao receber um ponto da lista_x retorna um ponto da lista_x
    '''

    func_dicio = dict()
    for x,y in zip(lista_x,lista_y):
        func_dicio[x] = y 
    
    def func(pont):
        if pont in func_dicio.keys():
            return func_dicio[pont]
        else:
            if exception == None:
                raise Exception('Esse ponto não foi definido na função')
            else:
                return exception

    return func


def duplicate(lista) -> list:
    '''
    Duplica cada elemento da lista e mantem a ordem, necessaria para o calculo por exemplo da interpolação de hermite,
    recebe: [1,2,3,4] e retorna: [1,1,2,2,3,3,4,4]
    '''


    l = []
    for i in lista:
        l.append(i)
        l.append(i)
    return l


def ddfunc(Point_list:list,derivada,func)-> list:
    '''
    Recebe a lista de pontos, uma função que retorna as derivadas em cada ponto, e a função que queremos usar na interpolação,
    e retorna as f[] necessarias para o calculo da intepolação de hermite em ordem,
    por exemplo [f[x_0],f[x_0,x_0],f[x_0,x_0,x_1],...] .
    '''
    subslist1,subslist2 = Point_list.copy(),Point_list.copy()#sublist1 e sublist2 são listas que usarei para guardar quais valores serão subtraidos nos denomidaores
    Point_list = [func(p) for p in Point_list] #aplica na lista de pontos a função e retorna cada valor
    
    def der(P_list): #funciona com uma redução de lista, seja x_i o elemento da nova lista e x1_i o elemento da lista antiga de posição i, x_i = (x1_(i+1) - x1_i)/(sublist[i]-sublist2[i]), da mesma forma que seria calcular a interpolação por tabela,  
        new_list = [] #salva nessa lista
        subslist1.pop(0)
        subslist2.pop()
        for i in range(len(P_list)-1):
            if subslist1[i] == subslist2[i]:
                new_list.append(derivada(subslist1[i]))
            else:
                new_list.append((P_list[i+1] - P_list[i])/(subslist1[i] - subslist2[i]))

        return new_list

    result_list = []
    while len(Point_list) != 1: #vai reduzindo a lista até sobrar apenas um elemento, e guarda apenas o topo na tabela, no caso o primeiro da lista
        result_list.append(Point_list[0]) 
        Point_list = der(Point_list)
    result_list.append(Point_list[0])
    return result_list

def interpolação_de_hermite(x,y,plot:bool = False,grid:bool = True):
    '''Essa função retorna, recebendo uma lista de valores de x e outra lista dos respectivos valores de f(x), uma função que interpola valores conforme f(x)
    
    Parâmetros:
        x (float): Lista de valores de x.
        y (float): Lista de valores conhecidos de f(x).
        (Ambos devem estar em ordem)
        plot (bool): Determina se a função deve plotar as informações ou não
        grid (bool): Determina se a função for plotar se a plotagem deve ter grid ou não
        
    Retorna:
        Função: Essa nova função recebe valores de x e retorna os valores de f(x) interpolados.
    
    
    '''

    f = function_definer(x,y) #define a função que f(x) = y
    d = diff_numerica(zip(x,y)) #gera a lista de derivadas de cada ponto de x
    f_linha = function_definer(x,d,exception=0) # define a função 'derivada' f'(x) = y'
    x_duplicated = duplicate(x)#prepara a lista para obter os coeficientes da função
    coeficientes_hermite = ddfunc(x_duplicated,f_linha,f)#calcula os resultados dos f[x_0],f[x_0,x_0] ... necessários
    
    def interpolation(ponto,plt = plot): #função que será retornada
        '''
        Essa função esta diretamente ligada a função original interpolação_de_hermite que a gerou, e seu resultado depende diretamente da função original


        Parâmetros:
            ponto (float): ponto x que será calculado o f(x)
            plt: Determina se a função deve ou não plotar o ponto x (só funciona se na função interpolação de hermite o parametro: plot = True)
        
        Retorna:
            float: Valor f(x)
        '''
        
        
        soma = 0 #para os (x - x_i), que crecem assim como os pontos da lista x_duplicate
        hermite = 0 
        for i in coeficientes_hermite:# para cadaf[x_i]
            mult = 1
            for j in x_duplicated[:soma]:#calcula os (x - x_i)
                mult = mult*(ponto - j)


            hermite += mult*i # + f[x_0,...,x_i]*(x-x_0)^2 * ... (x - x_i)
            soma += 1
        
        if plt:
            ax.scatter([ponto],[hermite],color = 'red')





        return hermite

    if plot:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.scatter(x,y,color = 'Black',label = 'Pontos Originais')
            
            a = min(x)
            b = max(x)
            c = 1
            d = 100
            e = 0
            dist = b-a

            if dist < 1:
                print('A distância entre os pontos está bem pequena')
                c =10
                d = 1
                e = 1
                while (b-a)*c<10:
                    c = c*10
            
            cred =lambda x: x/c
            xval = np.linspace(int(a*c)-e,int(b*c)+e,int((b-a)*c)*d)
            xval = np.array(list(map(cred,xval)))
            plot = False
            yval = np.array(list(map(lambda x: interpolation(x,plt=False),xval))) 
            ax.plot(xval,yval)
            ax.set_aspect('equal')
            ax.grid(grid)


    return interpolation




if __name__ == '__main__':
    print(diff_numerica.__doc__)
    print(ddfunc.__doc__)
    print(duplicate.__doc__)



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













            