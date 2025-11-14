import sys
import os
notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from CB2325NumericaG07.erros import erro_absoluto, erro_relativo

# TESTES DO ERRO ABSOLUTO
def test_erro_absoluto_basico():
    assert erro_absoluto(10, 8) == 2
    assert erro_absoluto(5, 8) == 3
    assert erro_absoluto(-3, 1) == 4


def test_erro_absoluto_zero():
    assert erro_absoluto(0, 0) == 0
    assert erro_absoluto(0, 5) == 5

# TESTES DO ERRO RELATIVO
def test_erro_relativo_basico():
    assert erro_relativo(10, 8) == 0.2   
    assert erro_relativo(5, 4) == 0.2
    assert erro_relativo(-4, -3) == 0.25  

def test_erro_relativo_zero_zero():
    # 0/0 => definido como 0.0 na função
    assert erro_relativo(0, 0) == 0.0

def test_erro_relativo_real_zero_aprox_diferente():
    # valor_real = 0 e valor_aprox != 0 → erro infinito
    assert erro_relativo(0, 5) == float('inf')

def test_erro_relativo_normal():
    assert erro_relativo(100, 99) == 0.01

