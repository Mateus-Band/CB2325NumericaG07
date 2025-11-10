import matplotlib
import numpy as np
import pytest

from CB2325NumericaG07.aproximacao import *
from pytest import approx

matplotlib.use("Agg")

###########
# Testes - Ajuste Linear
###########

# Teste para tamanho inválido de listas

def test_ajuste_linear_invalid_input_leght():
    x = [1, 2, 3]
    y = [1, 2]

    with pytest.raises(
        ValueError, match="As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho."
    ):
        ajuste_linear(x, y, plt_grafico=False)

# Teste para variância zero

def test_ajuste_linear_zero_variance():
    x = [2, 2, 2, 2]
    y = [1, 3, 5, 2]

    with pytest.raises(
        ValueError, match="A variância de valores_x é zero."
    ):
        ajuste_linear(x, y, plt_grafico=False)

# Teste para função ajustada

def test_ajuste_linear_perfeito():
    x = [1, 2, 3, 4, 5]
    y = [3, 5, 7, 9, 11] 
    a, b = ajuste_linear(x, y, plt_grafico=False)

    assert a == approx(2.0)
    assert b == approx(1.0)

# Teste com ruído

def test_ajuste_linear_ruido():
    x = [0, 1, 2, 3]
    y = [1, 2.1, 2.9, 4] 
    a, b = ajuste_linear(x, y, plt_grafico=False)

    assert a == approx(0.98)
    assert b == approx(1.03)

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
        + rng.normal(0, 1, size=x.shape)
    )

    coeffs = ajuste_polinomial(x, y, 2, plt_grafico=False)

    # Reconstrói Y com os coeficientes estimados

    y_fit = coeffs[0] + coeffs[1] * x + coeffs[2] * x ** 2

    # Mede o RMS

    rms = np.sqrt(np.mean((y_fit - y) ** 2))
    assert rms < 1

###########
# Testes - Ajuste Senoidal
###########

def test_ajuste_senoidal():

    # Sem ruído e com Gráfico - Frequência Positiva

    A1, B1, C1, D1 = 3.5, 0.5, 1.5, 3
    x1 = np.linspace(-15 * np.pi, 15 * np.pi, 200)
    y1 = A1 * np.sin(B1 * x1 + C1) + D1

    x1_lista = x1.tolist()
    y1_lista = y1.tolist()

    coeffs1 = ajuste_senoidal(x1_lista, y1_lista, 
                T_aprox=12.5, plt_grafico=True)
    A1_e, B1_e, C1_e, D1_e = coeffs1

    # Reconstrói y com os coeficientes estimados

    x1_arr = np.array(x1_lista)
    y1_fit = A1_e * np.sin(B1_e * x1_arr + C1_e) + D1_e

    # Verifica se está retornando 4 coeficientes

    assert len(coeffs1) == 4

    # Mede o RMS

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

    # Mede o RMS

    rms2 = np.sqrt(np.mean((y2_fit - y2) ** 2))
    assert rms2 < 0.25

    # Com ruído - Frequência Positiva

    rng = np.random.default_rng(42)
    A3, B3, C3, D3 = 3.5, 0.5, 1.5, 3
    x3 = np.linspace(-15 * np.pi, 15 * np.pi, 200)
    y3 = A3 * np.sin(B3 * x3 + C3) + D3 + rng.normal(0, 1, size=x3.shape)

    x3_lista = x3.tolist()
    y3_lista = y3.tolist()

    coeffs3 = ajuste_senoidal(x3_lista, y3_lista, T_aprox=12.5, plt_grafico=False)
    A3_e, B3_e, C3_e, D3_e = coeffs3

    # Reconstrói y com os coeficientes estimados

    x3_arr = np.array(x3_lista)
    y3_fit = A3_e * np.sin(B3_e * x3_arr + C3_e) + D3_e

    # Mede o RMS

    rms3 = np.sqrt(np.mean((y3_fit - y3) ** 2))
    assert rms3 < 1

###########
# Testes - Ajuste Exponencial
###########

# Teste para tamanho de lista inválidos

def test_ajuste_exponencial_invalid_input_lenght():
    x = [1, 2, 3]
    y = [1, 2]

    with pytest.raises(
        ValueError, match="As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho."
    ):
        ajuste_exponencial(x, y, plt_grafico=False)

# Teste para y não positivo

def test_ajuste_exponencial_y_nao_positivo():
    x = [1, 2, 3, 4]
    y = [1, 4, -9, 16]

    with pytest.raises(
        ValueError, match="A lista de valores de y possui valor\\(es\\) não postivos\\(s\\)."
    ):
        ajuste_exponencial(x, y, plt_grafico=False)

# Teste para função ajustada

def test_ajuste_exponencial_perfeito():
    x = np.array([0, 1, 2, 3, 4])
    y = 0.5 * np.exp(5 * x)
    
    a, b = ajuste_exponencial(x.tolist(), y.tolist(), plt_grafico=False)
    
    assert a == approx(5)
    assert b == approx(0.5)

# Teste para dados com ruído

def test_ajuste_exponencial_ruido():
    x = [0, 1, 2]
    y = [3.069674, 3.730489, 4.215573]

    a, b = ajuste_exponencial(x, y, plt_grafico=False)
    
    assert a == approx(0.16, rel=0.2)
    assert b == approx(3.1, rel=0.2)

###########
# Testes - Ajuste Logaritmo
###########

# Teste para tamanho de lista inválidos

def test_ajuste_logaritmo_invalid_input_lenght():
    x = [1, 2, 3]
    y = [1, 2]

    with pytest.raises(
        ValueError, match="As listas 'valores_x' e 'valores_y' devem ter o mesmo tamanho."
    ):
        ajuste_logaritmo(x, y, plt_grafico=False)

# Teste para x não positivo

def test_ajuste_logaritmo_x_nao_positivo():
    x = [1, 2, -3, 4]
    y = [1, 4, 9, 16]

    with pytest.raises(
        ValueError, match="A lista de valores de x possui valor\\(es\\) não postivos\\(s\\)."
    ):
        ajuste_logaritmo(x, y, plt_grafico=False)

# Teste para função ajustada

def test_ajuste_logaritmo_perfeito():
    x = np.array([1, 2, 3, 4, 5])
    y = 5 + 2 * np.log(x)
    
    a, b = ajuste_logaritmo(x.tolist(), y.tolist(), plt_grafico=False)
    
    assert a == approx(5)
    assert b == approx(2)

# Teste para dados com ruído

def test_ajuste_logaritmo_ruido():
    x = [1, 2, 3, 4, 5]
    y = [10.1, 12.0, 13.5, 14.0, 14.9]

    a, b = ajuste_logaritmo(x, y, plt_grafico=False)
    
    assert a == approx(10, rel=0.1)
    assert b == approx(3, rel=0.1)

###########
# Testes - Ajuste Múltiplo
###########

@pytest.mark.parametrize(
    "expected1",
    [
        np.array([1, -5, -6, 0]),
        np.array([1, 3, 4, 7]),
        np.array([0.45, 2.1, 50, 99]),
        np.array([-2, -2, -2, -2]),
        np.array([0, 0, 0, 0]),
        np.array([2.5, 3.1, 5.7, -2.7]),
        np.array([7.52, 8, 31.04, 500]),
    ],
)
def teste_ajuste_multiplo(expected1):

    # Teste para 'x_matriz' e 'z_matriz' de tamanhos diferentes

    a = np.array([9, -13, 5.5, -213, 44.95])
    b = np.array([15, 33, 94, -0.5, 0.88])
    z = np.array([1, 2])

    with pytest.raises(
        ValueError, match=(
            "Número de linhas de X e número de "
            "valores de Z não coincidem."
        )
    ):
        ajuste_multiplo([a, b], z, incluir_intercepto=False)

    # Teste para colinearidade

    a = np.array([9, -13, 5.5, -213, 44.95])
    b = np.array([15, 33, 94, -0.5, 0.88])
    c = np.array([-24, -24, -24, -24, -24])
    d = np.array([-48, -48, -48, -48, -48])

    z = (
        expected1[0] * a + 
        expected1[1] * b + 
        expected1[2] * c +
        expected1[3] * d
    )

    with pytest.raises(
        ValueError, match=(
            "Colinearidade detectada por matriz mal-condicionada. "
            "Verifique os dados de entrada."
        )
    ):
        ajuste_multiplo([a, b, c, d], z, incluir_intercepto=False)

    # Teste para dados sem ruído e sem intercepto

    a = np.array([9, -13, 5.5, -213, 44.95])
    b = np.array([15, 33, 94, -0.5, 0.88])
    c = np.array([-24, -24, -24, -24, -24])
    d = np.array([33, -3.22, -178, -26, 500])

    z = (
        expected1[0] * a + 
        expected1[1] * b + 
        expected1[2] * c +
        expected1[3] * d
    )

    result = ajuste_multiplo([a, b, c, d], z, incluir_intercepto=False)
    assert result == approx(expected1, rel=1e-4, abs=1e-5)

    # Teste para dados sem ruído e com intercepto

    a = np.array([9, -13, 5.5, -213, 44.95])
    b = np.array([15, 33, 94, -0.5, 0.88])
    c = np.array([33, -3.22, -178, -26, 500])

    z = (
        expected1[0] * a + 
        expected1[1] * b + 
        expected1[2] * c + 100
    )

    result = ajuste_multiplo([a, b, c], z)
    assert result == approx(
        [100, expected1[0], expected1[1], expected1[2]], rel=1e-4, abs=1e-5)
    
    # Teste para dados com ruído e com intercepto

    rng = np.random.default_rng(42)
    a = np.array([9, -13, 5.5, -213, 44.95])
    b = np.array([15, 33, 94, -0.5, 0.88])
    c = np.array([33, -3.22, -178, -26, 500])

    z = (
        expected1[0] * a + 
        expected1[1] * b + 
        expected1[2] * c + 100
        + rng.normal(0, 1, size=a.shape)
    )

    coeffs = ajuste_multiplo([a, b, c], z)
    I, A, B, C = coeffs

    # Reconstrói Z com os coeficientes estimados

    z_fit = A * a + B * b + C * c + I

    rms3 = np.sqrt(np.mean((z_fit - z) ** 2))
    assert rms3 < 1

###########
# Testes - Melhor Ajuste
###########

@pytest.mark.parametrize(
    "criterio",
    ["R2", "R2A", "AIC", "AICc", "BIC"])

def test_melhor_ajuste_linear(criterio):

    # Dados gerados a partir de um modelo linear

    rng = np.random.default_rng(42)
    x = np.linspace(-10, 8, 30)
    y = 5 * x + 8 + rng.normal(0, 0.5, size=x.shape)

    mod, info = melhor_ajuste(x, y, criterio)

    if criterio == "R2":
        pytest.skip("R2 tende a favorecer modelos mais complexos, " \
        "gerando overfitting")

    assert "linear" in mod

    for key in ["R2", "R2A", "AIC", "AICc", "BIC"]:
        assert key in info

@pytest.mark.parametrize(
    "criterio",
    ["R2", "R2A", "AIC", "AICc", "BIC"])

def test_melhor_ajuste_polinomial(criterio):

    # Dados gerados a partir de um modelo linear

    rng = np.random.default_rng(42)
    x = np.linspace(-20, 34, 60)
    y = 5 * x ** 3 + 8 * x + rng.normal(0, 1, size=x.shape)

    mod, info = melhor_ajuste(x, y, criterio)

    if criterio == "R2":
        pytest.skip("R2 tende a favorecer modelos mais complexos, " \
        "gerando overfitting")

    assert "polinomial grau 3" in mod

    for key in ["R2", "R2A", "AIC", "AICc", "BIC"]:
        assert key in info