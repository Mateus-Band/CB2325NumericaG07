import sys
import os
notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


import numpy as np
import matplotlib 
import pytest
from pytest import approx

matplotlib.use('Agg')

from CB2325NumericaG07.integracao import *




def test_integral_linear():
    """Testa a integral de uma função linear f(x) = x."""
    np.random.seed(42) 
    f_linear = lambda x: x
    
    # Integral de x, de 0 a 2 = 2
    # (4/2 - 0/2 = 2)
    resultado = integral_de_montecarlo(f_linear, a=0, b=2, qte=20000)
    

    assert resultado == approx(2.0, abs=0.1)

def test_integral_simetrica_seno():
    """Testa a integral de sin(x) em um período completo [0, 2*pi]."""
    np.random.seed(42)
    f_seno = lambda x: np.sin(x)
    
    # Integral de sin(x) de 0 a 2*pi = 0
    resultado = integral_de_montecarlo(f_seno, a=0, b=2*np.pi, qte=40000)
    
    # A área líquida deve ser próxima de 0
    assert resultado == approx(0.0, abs=0.1)

def test_integral_quadratica_seno():
    """
    Testa a integral f(x) = x^2 * sin(x) de 0 a 10.
    Valor : -98*cos(10) + 20*sin(10) - 2 ≈ 69.348
    """
    np.random.seed(42)
    f_complexa = lambda x: (x**2) * np.sin(x)
    
    valor_real = -98 * np.cos(10) + 20 * np.sin(10) - 2 # ~69.348
    
    # varios pontos para maior precisão
    resultado = integral_de_montecarlo(f_complexa, a=0, b=10, qte=100000)
    
    assert resultado == approx(valor_real, abs=1.5)

def test_integral_qte_zero():
    """
    Testa o comportamento de borda com qte=0.
    Deve evitar uma ZeroDivisionError e retornar 'nan'.
    """
    f_const = lambda x: 5
    
    with pytest.warns(RuntimeWarning, match='qte não pode ser igual a 0'):
        resultado = integral_de_montecarlo(f_const, a=0, b=10, qte=0)
    
    assert np.isnan(resultado)

