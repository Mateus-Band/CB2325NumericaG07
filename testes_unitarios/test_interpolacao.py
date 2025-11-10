import pytest
import sympy as sp
import numpy as np
import sys
import os
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
