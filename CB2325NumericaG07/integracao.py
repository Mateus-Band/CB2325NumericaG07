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

    #preparação para plotagem
    '''if plotar:
        def plot():'''
        


    a = float(a)
    b = float(b)
    h = (b - a) / n

    def x(i):
        return a + i * h
    
    soma_interna = 0
    for i in range(1, n):
        soma_interna += function(x(i))

    # x0 é a e xn é b
    integral = (h/2) * (function(a) + 2 * soma_interna + function(b))

    return integral




if __name__ == "__main__":
    print(integral_trapezio.__doc__)

