import pytest
import sys
import os
import math
import numpy as np
import sympy as sp
import matplotlib
from pytest import approx

# Configuração de caminho para importar o módulo
notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from CB2325NumericaG07.integracao import (
    integral_trapezio,
    integral_simpson38,
    integral_boole,
    integral_gauss_legendre,
    integral_de_montecarlo
)

# Configura matplotlib para não abrir janelas durante os testes
matplotlib.use('Agg')

import warnings
warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
    category=UserWarning
)

###########
# Testes - Regra do Trapézio
###########

def test_trapezio_linear():
    """
    A regra do trapézio é exata para funções lineares.
    Integral de f(x) = 2x de 0 a 1 é 1.
    """
    func = lambda x: 2 * x
    resultado = integral_trapezio(func, 0, 1, n=10)
    assert resultado == approx(1.0)

def test_trapezio_sympy_expr():
    """
    Testa suporte a expressões SymPy.
    Integral de sin(x) de 0 a pi é 2.
    """
    x = sp.Symbol('x')
    expr = sp.sin(x)
    # n alto para boa precisão
    resultado = integral_trapezio(expr, 0, math.pi, n=1000)
    assert resultado == approx(2.0, rel=1e-5)

def test_trapezio_sympy_lambda():
    """Testa suporte a sp.Lambda."""
    x = sp.Symbol('x')
    lmb = sp.Lambda(x, x**2)
    # Integral x^2 [0, 1] = 1/3
    resultado = integral_trapezio(lmb, 0, 1, n=1000)
    assert resultado == approx(1/3, rel=1e-4)

def test_trapezio_erro_n():
    """Deve levantar ValueError se n < 1."""
    with pytest.raises(ValueError, match=r"O número de subintervalos \(n\) deve ser pelo menos 1"):
        integral_trapezio(lambda x: x, 0, 1, n=0)

def test_trapezio_erro_sympy_multiplas_vars():
    """Deve levantar TypeError para funções com múltiplas variáveis."""
    x, y = sp.symbols('x y')
    expr = x + y
    with pytest.raises(TypeError, match="Expressão SymPy com múltiplas variáveis não é suportada"):
        integral_trapezio(expr, 0, 1, n=10)

###########
# Testes - Simpson 3/8
###########

def test_simpson38_polinomio_cubico():
    """
    Simpson 3/8 é exato para polinômios de até grau 3.
    Integral de x^3 de 0 a 1 é 0.25.
    """
    func = lambda x: x**3
    # n deve ser múltiplo de 3
    resultado = integral_simpson38(func, 0, 1, n=3)
    assert resultado == approx(0.25)

def test_simpson38_erro_n():
    """Deve levantar ValueError se n não for múltiplo de 3."""
    with pytest.raises(ValueError, match="n deve ser múltiplo de 3"):
        integral_simpson38(lambda x: x, 0, 1, n=4)

###########
# Testes - Regra de Boole
###########

def test_boole_polinomio_grau_5():
    """
    Boole é exata para polinômios de grau até 5? (Geralmente grau 4, mas erro é O(h^7)).
    Vamos testar grau 4: x^4 de 0 a 1 = 0.2.
    """
    func = lambda x: x**4
    # n deve ser múltiplo de 4
    resultado = integral_boole(func, 0, 1, n=4)
    assert resultado == approx(0.2)

def test_boole_erro_n():
    """Deve levantar ValueError se n não for múltiplo de 4."""
    with pytest.raises(ValueError, match="n deve ser múltiplo de 4"):
        integral_boole(lambda x: x, 0, 1, n=5)

###########
# Testes - Gauss-Legendre
###########

def test_gauss_legendre_exato():
    """
    Gauss-Legendre com n pontos é exato para polinômios de grau 2n-1.
    Para n=3, deve ser exato até grau 5.
    Integral de x^4 de -1 a 1 = 2/5 = 0.4.
    """
    func = lambda x: x**4
    resultado = integral_gauss_legendre(func, -1, 1, n=3)
    assert resultado == approx(0.4)

def test_gauss_legendre_transcendental():
    """Teste com função transcendental (exponencial)."""
    # Integral de e^x de 0 a 1 = e - 1 ~= 1.71828
    func = lambda x: np.exp(x)
    resultado = integral_gauss_legendre(func, 0, 1, n=5)
    assert resultado == approx(math.e - 1, rel=1e-8)

###########
# Testes - Monte Carlo
###########

def test_montecarlo_area_retangulo():
    """
    Integral de f(x)=1 de 0 a 1 deve ser 1.
    Usamos semente fixa para reprodutibilidade.
    """
    np.random.seed(42)
    func = lambda x: np.ones_like(x)
    # Com 5000 pontos, deve convergir razoavelmente
    resultado = integral_de_montecarlo(func, 0, 1, qte=5000)
    assert resultado == approx(1.0, rel=0.1) # Margem de erro maior por ser estocástico

def test_montecarlo_erro_qte():
    """Deve levantar RuntimeWarning se qte for 0."""
    with pytest.raises(RuntimeWarning, match="qte não pode ser igual a 0"):
        integral_de_montecarlo(lambda x: x, 0, 1, qte=0)

###########
# Testes - Plotagem (Smoke Tests)
###########

def test_plot_smoke_test():
    """
    Verifica se as chamadas com plotar=True não geram exceções.
    Não verifica o gráfico visualmente, apenas a execução.
    """
    func = lambda x: x**2
    
    # Testando Trapezio plot
    try:
        integral_trapezio(func, 0, 1, n=10, plotar=True)
    except Exception as e:
        pytest.fail(f"Trapezio plot falhou: {e}")

    # Testando Simpson plot
    try:
        integral_simpson38(func, 0, 1, n=3, plotar=True)
    except Exception as e:
        pytest.fail(f"Simpson plot falhou: {e}")
    
    # Testando Boole plot
    try:
        integral_boole(func, 0, 1, n=4, plotar=True)
    except Exception as e:
        pytest.fail(f"Boole plot falhou: {e}")
    
    # Testando Gauss plot
    try:
        integral_gauss_legendre(func, 0, 1, n=3, plotar=True)
    except Exception as e:
        pytest.fail(f"Gauss plot falhou: {e}")
        
    # Testando Monte Carlo plot
    try:
        integral_de_montecarlo(lambda x: x, 0, 1, qte=100, plot=True)
    except Exception as e:
        pytest.fail(f"Monte Carlo plot falhou: {e}")