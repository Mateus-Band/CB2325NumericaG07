import matplotlib
import numpy as np
import pytest
import sympy as sp

from CB2325NumericaG07.aproximacao import *
from pytest import approx

matplotlib.use("Agg")

###########
# Testes - Ajuste Polinomial
###########

# Teste para tamanho inválido de listas

def test_invalid_input_length():
    x = np.linspace(0, 5, 10)
    y = np.linspace(2, 4, 9)

    with pytest.raises(
        ValueError, match="As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho."
    ):
        ajuste_polinomial(x, y, 2, plt_grafico=False)

# Teste para plot do gráfico

@pytest.mark.parametrize(
    "expected1",
    [
        np.array([1, -5, -6]),
        np.array([1, 3, 4]),
        np.array([6, 3, 5]),
        np.array([0, 0, 0]),
        np.array([2.5, 3.1, 5.7]),
        np.array([7.52, 8, 31.04]),
    ],
)
def test_plot_withouerror(expected1):
    x = np.linspace(-10, 10, 200)
    y = expected1[2] * x ** 2 + expected1[1] * x + expected1[0]
    modelo = ajuste_polinomial(x, y, 2)

# Teste para verificar se os coeficientes gerados correspondem ao grau

@pytest.mark.parametrize(
    "expected1",
    [
        np.array([1, -5, -6]),
        np.array([1, 3, 4]),
        np.array([6, 3, 5]),
        np.array([0, 0, 0]),
        np.array([2.5, 3.1, 5.7]),
        np.array([7.52, 8, 31.04]),
    ],
)
def test_poly_degree(expected1):
    degree_list = [0, 5, 4, 10]
    x = np.linspace(-10, 10, 20)
    y = expected1[2] * x ** 2 + expected1[1] * x + expected1[0]

    for degree in degree_list:
        modelo = ajuste_polinomial(x, y, degree, plt_grafico=False)
        dtst = degree + 1
        assert len(modelo) == dtst

# Teste para a lógica do código

@pytest.mark.parametrize(
    "expected2",
    [
        np.array([1, -5, -6, 3, 8, 10]),
        np.array([3, -8, -16, 9, 14, 11]),
        np.array([2, -0.6, -19, 10, 11, 12]),
        np.array([4, -4, -4, 4, 4, -4]),
        np.array([1, -10, -49, 0, 0, -5]),
        np.array([0, -5, -0, 3, 1, 0.3]),
    ],
)
def test_logic(expected2):

    x = np.linspace(-10, 10, 200)
    y = (
        expected2[5] * x ** 5
        + expected2[4] * x ** 4
        + expected2[3] * x ** 3
        + expected2[2] * x ** 2
        + expected2[1] * x
        + expected2[0]
    )

    result = ajuste_polinomial(x, y, 5, plt_grafico=False)

    assert result == approx(expected2, rel=1e-6, abs=1e-8)

# Teste para dados com ruídos

@pytest.mark.parametrize(
    "expected1",
    [
        np.array([1, -5, -6]),
        np.array([1, 3, 4]),
        np.array([6, 3, 5]),
        np.array([-2, -2, -2]),
        np.array([2.5, 3.1, 5.7]),
        np.array([7.52, 8, 31.04]),
    ],
)
def test_werror(expected1):

    rng = np.random.default_rng(42)
    x = np.linspace(-10, 10, 200)
    y = (
        expected1[2] * x ** 2
        + expected1[1] * x
        + expected1[0]
        + rng.normal(0, 0.8, size=x.shape)
    )

    result = ajuste_polinomial(x, y, 2, plt_grafico=False)
    assert result == approx(expected1, rel=0.1, abs=1e-4)

###########
# Teste - Ajuste Senoidal
###########

def test_ajuste_senoidal_basico():

    # Sem ruído - Frequência Positiva

    A1, B1, C1, D1 = 3.5, 0.5, 1.5, 3
    x1 = np.linspace(-15 * np.pi, 15 * np.pi, 200)
    y1 = A1 * np.sin(B1 * x1 + C1) + D1

    x1_lista = x1.tolist()
    y1_lista = y1.tolist()

    coeffs1 = ajuste_senoidal(x1_lista, y1_lista, T_aprox=12.5, plt_grafico=False)
    A1_e, B1_e, C1_e, D1_e = coeffs1

    # Reconstrói y com os coeficientes estimados
    x1_arr = np.array(x1_lista)
    y1_fit = A1_e * np.sin(B1_e * x1_arr + C1_e) + D1_e

    assert len(coeffs1) == 4
    rms1 = np.sqrt(np.mean((y1_fit - y1) ** 2))
    assert rms1 < 0.1
    # Sem ruído - Frequência Negativa

    A2, B2, C2, D2 = 4, -2.5, -0.5, 0
    x2 = np.linspace(-15 * np.pi, 15 * np.pi, 200)
    y2 = A2 * np.sin(B2 * x2 + C2) + D2

    x2_lista = x2.tolist()
    y2_lista = y2.tolist()

    coeffs2 = ajuste_senoidal(x2_lista, y2_lista, T_aprox=2.5, plt_grafico=False)
    A2_e, B2_e, C2_e, D2_e = coeffs2

    # Reconstrói y com os coeficientes estimados
    x2_arr = np.array(x2_lista)
    y2_fit = A2_e * np.sin(B2_e * x2_arr + C2_e) + D2_e

    rms2 = np.sqrt(np.mean((y2_fit - y2) ** 2))
    assert rms2 < 0.25

    # Com ruído

    rng = np.random.default_rng(42)
    A3, B3, C3, D3 = 3.5, 0.5, 1.5, 3
    x3 = np.linspace(-15 * np.pi, 15 * np.pi, 200)
    y3 = A3 * np.sin(B3 * x3 + C3) + D3 + rng.normal(0, 1, size=x2.shape)

    x3_lista = x3.tolist()
    y3_lista = y3.tolist()

    coeffs3 = ajuste_senoidal(x3_lista, y3_lista, T_aprox=12.5, plt_grafico=False)
    A3_e, B3_e, C3_e, D3_e = coeffs3

    # Reconstrói y com os coeficientes estimados
    x3_arr = np.array(x3_lista)
    y3_fit = A3_e * np.sin(B3_e * x3_arr + C3_e) + D3_e

    rms3 = np.sqrt(np.mean((y3_fit - y3) ** 2))
    assert rms3 < 1

