import matplotlib.pyplot as plt
import sympy as sp

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


if __name__ == "__main__":
    print(integral_trapezio.__doc__)

