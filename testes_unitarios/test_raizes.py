from unittest.mock import patch
import sys
import os
import pytest
import math

notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from CB2325NumericaG07.raizes import *

@pytest.fixture
def func_quadratica():
    return lambda x: x**2 - 4

@pytest.fixture
def func_cubica():
    return lambda x: x**3 - 9*x + 5

# Testes para o Método da Bisseção

def test_bissecao_sucesso_1(func_quadratica):
    """Testa se encontra a raiz 2.0 para x^2 - 4 no intervalo [0, 3]."""
    raiz = metodo_bissecao(func_quadratica, 0, 3, tol=1e-6)
    assert raiz == pytest.approx(2.0, abs=1e-5)

def test_bissecao_sucesso_2(func_cubica):
    """Testa se encontra 0.57688 para o exemplo: x^3 - 9x + 5."""
    raiz = metodo_bissecao(func_cubica, 0, 2, tol=1e-6)
    assert raiz == pytest.approx(0.57688, abs=1e-4)

def test_bissecao_erro_sinais_iguais(func_quadratica):
    """Testa se levanta ValueError quando f(a) e f(b) têm o mesmo sinal."""
    with pytest.raises(ValueError) as erro:
        metodo_bissecao(func_quadratica, 3, 5, tol=1e-6)
    assert "sinais opostos" in str(erro.value)

def test_bissecao_nao_convergencia(func_cubica):
    """Testa se retorna None quando atinge max_inter."""
    resultado = metodo_bissecao(func_cubica, 0, 2, tol=1e-6, max_inter=1)
    assert resultado is None

# Testes para o Método da Secante

def test_secante_sucesso_basico(func_quadratica):
    """Testa o exemplo da função quadrada com o método da secante"""
    raiz = metodo_secante(func_quadratica, 0, 3, tol=1e-6)
    assert raiz == pytest.approx(2.0, abs=1e-5)

def test_secante_exemplo_pdf(func_cubica):
    """Testa o exemplo da função cubica com o método da secante."""
    raiz = metodo_secante(func_cubica, 0, 2, tol=1e-6)
    assert raiz == pytest.approx(0.57688, abs=1e-4)

def test_secante_divisao_zero():
    """Testa o erro quando f(x0) == f(x1) (reta horizontal)."""
    f = lambda x: x**2
    with pytest.raises(ValueError) as erro:
        metodo_secante(f, -1, 1, tol=1e-6)
    assert "divisão por zero" in str(erro.value)

def test_secante_nao_convergencia(func_cubica):
    """Testa se retorna None quando atinge max_inter."""
    resultado = metodo_secante(func_cubica, 0, 2, tol=1e-6, max_inter=1)
    assert resultado is None

def test_secante_raiz_exata():
    """Testa o caso de sorte onde um dos pontos iniciais já é a raiz."""
    f = lambda x: x**2 - 1
    raiz = metodo_secante(f, 0.5, 1.0, tol=1e-6)
    assert raiz == 1.0

#Testes método newton_raphson

def test_nr_raiz_simples_polinomio():
    '''Testa para um polinomio simples de segundo grau'''
    funcao = lambda x: x**2 - 4
    raiz = metodo_newton_raphson(funcao, estimativa_inicial=3.0)
    assert raiz == pytest.approx(2.0)

def test_nr_raiz_simples_polinomio_negativa():
    '''Testa a raíz negativa de um polinomio simples de segundo grau'''
    funcao = lambda x: x**2 - 4
    raiz = metodo_newton_raphson(funcao, estimativa_inicial=-1.0)
    assert raiz == pytest.approx(-2.0)

def test_nr_raiz_transcendental():
    '''Testa para uma função transcendental'''
    funcao = lambda x: math.cos(x) - x
    raiz_conhecida = 0.739085133
    raiz = metodo_newton_raphson(funcao, estimativa_inicial=1.0)
    assert raiz == pytest.approx(raiz_conhecida, abs=1e-6)

def test_nr_estimador_interno_automatico():
    '''Testa se a estimativa inicial automatica está funcionando'''
    funcao = lambda x: (x - 5)**3
    raiz = metodo_newton_raphson(funcao, estimativa_inicial=None)
    assert raiz == pytest.approx(5.0, abs=1e-5)

def test_nr_nao_convergencia_max_iter(capsys):
    '''Testa o método nr para quando o limite de iterações é atingido'''
    funcao = lambda x: x**3 - 2*x + 2
    raiz = metodo_newton_raphson(funcao, estimativa_inicial=0.0, max_iter=10)
    out, err = capsys.readouterr()
    assert "O método não convergiu" in out

def test_nr_derivada_nula(capsys):
    """
    Testa Newton-Raphson: derivada nula com uma função constante f(x) = 5.
    """
    funcao = lambda x: 5
    estimativa = 3.0  
    raiz = metodo_newton_raphson(funcao, estimativa_inicial=estimativa, tol=1e-8)
    
    out, err = capsys.readouterr()
    assert "Derivada nula" in out
    
    assert raiz == estimativa

@patch('matplotlib.pyplot.show')
def test_nr_plotagem_executa_sem_erro(mock_show):
    '''Testa se o gráfico é executado sem erro'''
    funcao = lambda x: x - 1
    raiz = metodo_newton_raphson(funcao, estimativa_inicial=5.0, plotar=True)
    assert mock_show.called
    assert raiz == pytest.approx(1.0)





