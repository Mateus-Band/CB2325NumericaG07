import pytest
import sympy as sp
import numpy as np
import sys
import os
import matplotlib
from pytest import approx



notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


from CB2325NumericaG07.interpolacao import *


def test_interpolacao_linear_por_partes():
    """
    Testa a função interpolacao_linear_por_partes do módulo interpolacao.py
    verificando:
    - se os polinômios são construídos corretamente;
    - se o valor interpolado está correto;
    - e se há mensagem adequada quando x_test está fora do intervalo.
    """

    # Entradas
    x_vals = [0, 1, 2]
    y_vals = [1, 3, 2]
    x_test = 1.5

    # Valor esperado
    valor_esperado = 2.5

    # Chamada da função
    lista_poli, valor_teste = interpolacao_linear_por_partes(x_vals, y_vals, plotar=False, x_test=x_test)

    # Teste 1: tipos
    assert isinstance(lista_poli, list)
    assert isinstance(valor_teste, (int, float))

    # Teste 2: número de polinômios
    assert len(lista_poli) == len(x_vals) - 1

    # Teste 3: conteúdo do primeiro polinômio
    x = sp.Symbol('x')
    poli_esperado = 1 + 2 * (x - 0)  # f(x)=1+2x
    assert sp.simplify(lista_poli[0] - poli_esperado) == 0

    # Teste 4: valor interpolado
    assert np.isclose(valor_teste, valor_esperado, atol=1e-8)

    # Teste 5: fora do intervalo
    _, fora = interpolacao_linear_por_partes(x_vals, y_vals, x_test=10)
    assert isinstance(fora, str) and "Fora do intervalo" in fora





matplotlib.use('Agg')

###########
# Testes - Diferenciação Numérica (diff_numerica)
###########

def test_diff_numerica_nula():
    """Teste para função constante (derivada deve ser 0)."""
    x = [1, 2, 3, 4, 5]
    y = [2, 2, 2, 2, 2]
    d = diff_numerica(x, y)
    assert d == [0, 0, 0, 0, 0]

def test_diff_numerica_linear():
    """Teste para função linear y = 2x + 5 (derivada deve ser 2)."""
    x = [0, 1, 2, 3, 4]
    y = [5, 7, 9, 11, 13]
    d = diff_numerica(x, y)
    assert d == approx([2, 2, 2, 2, 2])

def test_diff_numerica_quadratica():
    """
    Teste para y = x^2.
    Esperado: [3,4,6,8,10,11]
    
    """
    x = [1, 2, 3, 4, 5, 6]
    y = [1, 4, 9, 16, 25, 36]
    d = diff_numerica(x, y)
    
    assert d == approx([3, 4, 6, 8, 10, 11])

def test_diff_numerica_invalid_input():
    """Teste para verificar erro com listas de tamanhos diferentes."""
    x = [1, 2, 3, 4]
    y = [1, 2, 3]

    with pytest.raises(ValueError, match='As listas x e y devem ter o mesmo tamanho'):
        diff_numerica(x, y)

###########
# Testes - Definidor de Função (function_definer)
###########

def test_function_definer_basic():
    """Teste básico de mapeamento."""
    lx = [1, 2, 3]
    ly = [10, 20, 30]
    f = function_definer(lx, ly)

    assert f(1) == 10
    assert f(2) == 20
    assert f(3) == 30

def test_function_definer_error_default():
    """Teste se levanta a exceção padrão para pontos não definidos."""
    lx = [1, 2]
    ly = [10, 20]
    f = function_definer(lx, ly)

    with pytest.raises(Exception, match='Esse ponto não foi definido na função'):
        f(3)

def test_function_definer_custom_exception_return():
    """Teste se retorna o valor de exceção personalizado."""
    lx = [1, 2]
    ly = [10, 20]
    f = function_definer(lx, ly, exception=np.nan)

    assert np.isnan(f(99))

def test_function_definer_invalid_input_length():
    """Teste de validação de tamanho de entrada."""
    
    with pytest.raises(ValueError, match='As listas x e y devem ter o mesmo tamanho'):
         function_definer([1, 2, 3], [1, 2])

###########
# Testes - Funções Auxiliares (duplicate)
###########

def test_duplicate_basic():
    entrada = [1, 2, 3]
    esperado = [1, 1, 2, 2, 3, 3]
    assert duplicate(entrada) == esperado

def test_duplicate_empty():
    assert duplicate([]) == []

###########
# Testes - Diferenças Divididas de Newton (newton_ddfunc)
###########

def test_newton_ddfunc_linear():
    """
    Para f(x) = x, os coeficientes de Newton devem ser [x0, 1, 0, 0...]
    Pontos: (1,1), (2,2), (3,3)
    
    Esperado: [0,1,1]
    """
    x = [1, 2, 3]
    y = [1, 2, 3]
    f_linear = function_definer(x, y)
    
    dd = newton_ddfunc(x, f_linear)
    assert dd == approx([1, 1, 0])

def test_newton_ddfunc_quadratica():
    """
    Para f(x) = x^2 em [0, 1, 2]:
    y = [0, 1, 4]
   
    Esperado: [0, 1, 1]
    """
    x = [0, 1, 2]
    y = [0, 1, 4]
    f_quad = function_definer(x, y)

    dd = newton_ddfunc(x, f_quad)
    assert dd == approx([0, 1, 1])

###########
# Testes - Diferenças Divididas de Hermite (hermite_ddfunc)
###########

def test_hermite_ddfunc_cubica():
    """
    Teste usando f(x) = x^3 e f'(x) = 3x^2 
    
    Coeficientes esperados (topo da tabela): [1, 3, 4, 1]
    """
    x_orig = [1, 2]
    x_dup = duplicate(x_orig) # [1, 1, 2, 2]

    # Funções reais para teste
    func_cubo = lambda x: x**3
    deriv_cubo = lambda x: 3 * x**2
    
    dd_hermite = hermite_ddfunc(x_dup, deriv_cubo, func_cubo)
    
    assert dd_hermite == approx([1, 3, 4, 1])

def test_hermite_ddfunc_constante():
    """
    Para f(x) = 5, f'(x) = 0.
    Todos os DDs após o primeiro termo devem ser 0.
    """
    x_dup = [1, 1, 3, 3]
    func_const = lambda x: 5
    deriv_const = lambda x: 0

    dd_hermite = hermite_ddfunc(x_dup, deriv_const, func_const)
    assert dd_hermite == approx([5, 0, 0, 0])



###########
# Testes - Interpolação de Newton
###########

def test_newton_interpolation_linear():
    """
    Teste com função linear f(x) = 2x + 1.
    Pontos: (0, 1), (2, 5)
    """
    x = [0, 2]
    y = [1, 5]
    
    # Cria a função interpoladora
    P_newton = interpolacao_de_newton(x, y, plot=False)
    
    # Verifica nos pontos originais
    assert P_newton(0) == approx(1)
    assert P_newton(2) == approx(5)
    
    # Verifica em pontos intermediários
    assert P_newton(1) == approx(3)   # 2(1) + 1 = 3
    assert P_newton(0.5) == approx(2) # 2(0.5) + 1 = 2

def test_newton_interpolation_quadratic():
    """
    Teste com f(x) = x^2.
    Pontos: (-1, 1), (0, 0), (2, 4)
    """
    x = [-1, 0, 2]
    y = [1, 0, 4]
    P_newton = interpolacao_de_newton(x, y, plot=False)
    
    assert P_newton(1) == approx(1)   # 1^2 = 1
    assert P_newton(3) == approx(9)   # 3^2 = 9
    assert P_newton(-2) == approx(4)  # (-2)^2 = 4

def test_newton_interpolation_degree():
    """
    Teste para valor fora do intervalo de ponto (extrapolação).
    f(x) = x^3
    """
    x = [-1, 0, 1, 2]
    y = [-1, 0, 1, 8]
    P_newton = interpolacao_de_newton(x, y, plot=False)
    
    # Testa um ponto fora da amostra
    assert P_newton(3) == approx(27) # 3^3 = 27

###########
# Testes - Interpolação de Hermite
###########

def test_hermite_interpolation_linear():
    """
    Teste com função linear f(x) = 2x + 1.
    Pontos: (0, 1), (2, 5)
    """
    x = [0, 2]
    y = [1, 5]
    
    # Cria a função interpoladora
    P_hermite = interpolacao_de_hermite(x, y, plot=False)
    
    # Verifica nos pontos originais
    assert P_hermite(0) == approx(1)
    assert P_hermite(2) == approx(5)
    
    # Verifica em pontos intermediários
    assert P_hermite(1) == approx(3)   # 2(1) + 1 = 3
    assert P_hermite(0.5) == approx(2) # 2(0.5) + 1 = 2

def test_hermite_interpolation_quadratic():
    """
    Teste com f(x) = x^2.
    Pontos: (-1, 1), (0, 0), (2, 4)
    Polinômio interpolador deve ser exatamente x^2.
    """
    x = [-1, 0, 2]
    y = [1, 0, 4]
    P_hermite = interpolacao_de_hermite(x, y, plot=False)

    #verificando nos pontos originais
    assert P_hermite(-1) == approx(1) 
    assert P_hermite(0) == approx(0) 
    assert P_hermite(2) == approx(4) 
 
    
    assert P_hermite(1) == approx(1)   # 1^2 = 1
    assert P_hermite(3) == approx(9)   # 3^2 = 9
    assert P_hermite(-2) == approx(4)  # (-2)^2 = 4

def test_hermite_interpolation_degree():
    """
    Teste para valor fora do intervalo de ponto (extrapolação).
    f(x) = x^3
    """
    x = [-1, 0, 1, 2]
    y = [-1, 0, 1, 8]
    P_hermite = interpolacao_de_hermite(x, y, plot=False)
    
    # Testa um ponto fora da amostra
    assert P_hermite(3) == approx(27) # 3^3 = 27


def test_hermite_vs_newton():
    """
    Para os mesmos pontos, se a função for um polinômio de grau baixo,
    Newton e Hermite (mesmo com derivadas aproximadas) devem dar resultados
    próximos.
    """
    x = np.linspace(0, 5, 10)
    y = np.sin(x)
    
    P_newton = interpolacao_de_newton(x, y, plot=False)
    P_hermite = interpolacao_de_hermite(x, y, plot=False)
    
    ponto_teste = 2.5
    val_newton = P_newton(ponto_teste)
    val_hermite = P_hermite(ponto_teste)
    
   
    assert val_hermite == approx(val_newton, abs=1e-1)


@pytest.mark.parametrize("num, expected",[
    (7, 5.4), 
    (3, 8.8), 
    (1, 1.5), 
    (7, 9.2), 
    (1, 2.1), 
    (7, 3.7), 
    (2, 4.6), 
    (6, 7.9), 
    (4, 0.2), 
    (2, 6.5), 
    (3, 1.9), 
    (7, 4.3), 
    (6, 8.1), 
    (7, 2.5), 
    (0, 9.9), 
    (3, 7.6), 
    (4, 3.2), 
    (7, 5.8), 
    (8, 0.5), 
    (2, 8.3)
])
def hermite_and_newton_expected_dots(num,expected):
    #conjunto de pontos
    x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    y = [5.4, 8.8, 1.5, 9.2, 2.1, 3.7, 4.6, 7.9, 0.2, 6.5, 1.9, 4.3, 8.1, 2.5, 9.9, 7.6, 3.2, 5.8, 0.5, 8.3]
    
    
    h =interpolacao_de_hermite(x,y)
    
    assert h(num) == expected
    
    
    n = interpolacao_de_newton(x,y)

    assert n(num) == expected



### Testes da Interpolação de Lagrange e Vandermonde ###

# Casos de teste comuns para ambas as funções
@pytest.mark.parametrize("interp_func", [interpolacao_polinomial, interp_vand])
class TestInterpolacaoPolinomial:
    
    def test_interp_linear(self, interp_func):
        """
        Teste com uma função linear: f(x) = 2x + 1.
        Pontos: (0, 1), (2, 5). O polinômio deve ser P(x) = 2x + 1.
        """
        pontos = [(0, 1), (2, 5)]
        
        # O polinômio esperado é 2*x + 1
        x = sp.Symbol('x')
        poli_esperado = 2*x + 1
        
        P_x = interp_func(pontos, plotar=False)
        
        # O resultado deve ser uma expressão SymPy
        assert isinstance(P_x, sp.Expr)
        
        # O polinômio simplificado deve ser igual ao esperado
        assert sp.simplify(P_x - poli_esperado) == 0
        
        # Teste de avaliação em um ponto (por exemplo, x=1 -> 3)
        assert P_x.subs(x, 1) == 3


    def test_interp_quadratica(self, interp_func):
        """
        Teste com uma função quadrática: f(x) = x^2.
        Pontos: (-1, 1), (0, 0), (2, 4). O polinômio deve ser P(x) = x^2.
        """
        pontos = [(-1, 1), (0, 0), (2, 4)]
        
        # O polinômio esperado é x**2
        x = sp.Symbol('x')
        poli_esperado = x**2
        
        P_x = interp_func(pontos, plotar=False)
        
        assert sp.simplify(P_x - poli_esperado) == 0
        
        # Teste de avaliação em um ponto não amostrado (por exemplo, x=3 -> 9)
        assert P_x.subs(x, 3) == 9
        
        
    def test_interp_constante(self, interp_func):
        """
        Teste com uma função constante: f(x) = 5.
        Pontos: (1, 5), (5, 5), (10, 5). O polinômio deve ser P(x) = 5.
        """
        pontos = [(1, 5), (5, 5), (10, 5)]
        
        x = sp.Symbol('x')
        poli_esperado = sp.Integer(5) 
        
        P_x = interp_func(pontos, plotar=False)
        
        assert sp.simplify(P_x - poli_esperado) == 0
        
        # Teste de avaliação (deve ser 5)
        assert P_x.subs(x, 99) == 5

    
    def test_interp_lista_vazia(self, interp_func):
        """
        Testa a entrada com lista de pontos vazia.
        """
        pontos = []
        resultado = interp_func(pontos, plotar=False)
        
        assert isinstance(resultado, str)
        assert "Lista de pontos vazia" in resultado


    def test_interp_precisao_float(self, interp_func):
        """
        Teste com coeficientes flutuantes (seno). Verifica a precisão da interpolação.
        """
        pi_val = np.pi
        pontos = [(0, 0), (pi_val/2, 1), (pi_val, 0)]
        
        x = sp.Symbol('x')
        # Polinômio quadrático exato que passa pelos pontos: P(x) = -(4/pi^2) * x^2 + (4/pi) * x
        poli_esperado_expr = (-4/pi_val**2) * x**2 + (4/pi_val) * x

        P_x = interp_func(pontos, plotar=False)
        
        P_x_numerico = sp.lambdify(x, P_x, 'numpy')
        
        x_teste = pi_val / 4
        valor_interp = P_x_numerico(x_teste)
        valor_esperado = poli_esperado_expr.subs(x, x_teste).evalf()
        
        assert valor_interp == approx(valor_esperado, abs=1e-8)


# Teste Específico para Vandermonde (Pontos Duplicados)
def test_interp_vand_pontos_duplicados():
    """
    Teste específico para Vandermonde quando há pontos x duplicados,
    o que leva a uma matriz singular (det = 0).
    """
    pontos = [(1, 2), (1, 5), (3, 4)] # Ponto x=1 duplicado
    
    resultado = interp_vand(pontos, plotar=False)
    
    assert isinstance(resultado, str)
    assert "Matriz de Vandermonde é singular" in resultado


# Teste de Funcionalidade de Plotagem (garante que o código de plotagem é executado sem erro)
def test_interpolacao_polinomial_plot():
    """Testa se a função retorna o polinomio e executa o codigo de plotagem sem erro."""
    pontos = [(1, 1), (2, 4), (3, 9)]
    
    # Chame a função com plotar=True
    P_x = interpolacao_polinomial(pontos, plotar=True)
    
    # O retorno ainda deve ser a expressão simbólica correta (x^2)
    x = sp.Symbol('x')
    poli_esperado = x**2 
    assert sp.simplify(P_x - poli_esperado) == 0

def test_interp_vand_plot():
    """Testa se a função retorna o polinomio e executa o codigo de plotagem sem erro."""
    pontos = [(1, 1), (2, 4), (3, 9)]
    
    # Chame a função com plotar=True
    P_x = interp_vand(pontos, plotar=True)
    
    # O retorno ainda deve ser a expressão simbólica correta (x^2)
    x = sp.Symbol('x')
    poli_esperado = x**2 
    assert sp.simplify(P_x - poli_esperado) == 0