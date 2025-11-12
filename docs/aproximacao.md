# `aproximacao.py` - Módulo de Aproximação

Este módulo fornece ferramentas para calcular funções de aproximação de pontos.

## Dependências

Este módulo requer as seguintes bibliotecas Python para sua funcionalidade completa:

* **NumPy**: Para operações numéricas e arrays.
* **SymPy**: Para manipulação de expressões simbólicas.
* **Matplotlib**: Para a visualização gráfica (opcional).

## Funções

### `ajuste_linear`

Calcula os coeficientes de um ajuste linear (modelo $y = ax + b$) para um conjunto de dados (valores_x, valores_y) usando o Método dos Mínimos Quadrados (MMQ).

#### Parâmetros

* **`valores_x`** (ArrayLike): A variável independente (lista ou array NumPy de valores).
* **`valores_y`** (ArrayLike): A variável dependente (lista ou array NumPy de valores).
* **`plt_grafico`** (bool, opcional): Se `True`, exibe um gráfico do ajuste usando `matplotlib`. O padrão é `False`.
* **`expr`** (bool, opcional): Se `True`, imprime a expressão matemática encontrada. O padrão é `False`.

#### Retorna

* **`np.ndarray`**: Um array NumPy contendo os coeficientes `[a, b]` do modelo.

#### Fórmula Matemática

O ajuste linear $y = ax + b$ é encontrado minimizando a soma dos quadrados dos resíduos. Os coeficientes são calculados como:

$$
a = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{Cov(valores\_x, valores\_y)}{Var(valores\_x)}
$$
$$
b = \bar{y} - a \bar{x}
$$

Onde:
* $n$ = número de pontos.
* $\bar{x}$ = média dos valores de valores_x.
* $\bar{y}$ = média dos valores de valores_y.

---

### `ajuste_exponencial`

Calcula os coeficientes de um ajuste exponencial (modelo $y = b \cdot e^{ax}$) para um conjunto de dados (valores_x, valores_y). O método lineariza o modelo aplicando o logaritmo natural.

#### Parâmetros

* **`valores_x`** (ArrayLike): A variável independente.
* **`valores_y`** (ArrayLike): A variável dependente. **Requer que todos os valores de valores_y sejam positivos.**
* **`plt_grafico`** (bool, opcional): Se `True`, exibe um gráfico do ajuste. O padrão é `False`.
* **`expr`** (bool, opcional): Se `True`, imprime a expressão matemática encontrada. O padrão é `False`.

#### Retorna

* **`np.ndarray`**: Um array NumPy contendo os coeficientes `[a, b]` do modelo.

#### Fórmula Matemática

O modelo $y = b \cdot e^{ax}$ é linearizado tomando o logaritmo natural em ambos os lados:
$$
\ln(y) = \ln(b) + ax
$$
Definindo $Y' = \ln(y)$ e $b' = \ln(b)$, o problema é reduzido a um ajuste linear $Y' = b' + ax$. Os coeficientes $a$ e $b'$ são encontrados por MMQ.

O coeficiente $b$ é então recuperado por:
$$
b = e^{b'}
$$

Onde:
* $a$ e $b'$ são os coeficientes do ajuste linear $Y' = ax + b'$.

---

### `ajuste_logaritmo`

Calcula os coeficientes de um ajuste logarítmico (modelo $y = a + b \cdot \ln(x)$) para um conjunto de dados (valores_x, valores_y). O método lineariza o modelo usando $\ln(x)$ como a nova variável independente.

#### Parâmetros

* **`valores_x`** (ArrayLike): A variável independente. **Requer que todos os valores de valores_x sejam positivos.**
* **`valores_y`** (ArrayLike): A variável dependente.
* **`plt_grafico`** (bool, opcional): Se `True`, exibe um gráfico do ajuste. O padrão é `False`.
* **`expr`** (bool, opcional): Se `True`, imprime a expressão matemática encontrada. O padrão é `False`.

#### Retorna

* **`np.ndarray`**: Um array NumPy contendo os coeficientes `[a, b]` do modelo.

#### Fórmula Matemática

O modelo $y = a + b \cdot \ln(x)$ é linearizado pela substituição $X' = \ln(x)$.
O problema é reduzido a um ajuste linear:
$$
y = a + bX'
$$
Os coeficientes $a$ e $b$ são então encontrados diretamente pelo Método dos Mínimos Quadrados.

Onde:
* $a$ e $b$ são os coeficientes do ajuste linear $y = a + bX'$.

---

### `avaliar_ajuste`

Avalia a qualidade de um modelo de ajuste (previamente calculado) usando um ou mais critérios estatísticos.

#### Parâmetros

* **`valores_x`** (ArrayLike): Lista de valores da variável independente usados no ajuste.
* **`valores_y`** (ArrayLike): Lista de valores da variável dependente usados no ajuste.
* **`criterio`** (str): O critério de avaliação desejado. Opções: `"R2"`, `"R2A"`, `"AIC"`, `"AICc"`, `"BIC"`, ou `"all"` (para retornar todos).
* **`modelo`** (str): O nome do modelo que gerou os coeficientes. Opções: `"linear"`, `"polinomial"`, `"exponencial"`, `"logaritmo"`, `"senoidal"`.
* **`coeficientes`** (tuple | np.ndarray): Os coeficientes retornados pela função de ajuste correspondente.

#### Retorna

* **`float`** | **`tuple`**: O valor do critério solicitado (se `criterio` != `"all"`) ou uma tupla contendo (R2, R2A, AIC, AICc, BIC) (se `criterio` == `"all"`).

#### Fórmula Matemática

As métricas são baseadas na Soma dos Quadrados dos Resíduos (RSS) e na Soma dos Quadrados Total (RST).

$$
RSS = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \quad \text{(Soma dos Quadrados dos Resíduos)}
$$
$$
RST = \sum_{i=1}^{n} (y_i - \bar{y})^2 \quad \text{(Soma dos Quadrados Total)}
$$

**Critérios:**
* **R² (Coef. de Determinação):** $R^2 = 1 - \frac{RSS}{RST}$
* **R² Ajustado:** $R^2_{A} = 1 - \frac{(1 - R^2)(n - 1)}{n - k - 1}$
* **AIC (Critério de Akaike):** $AIC = n \cdot \ln\left(\frac{RSS}{n}\right) + 2k$
* **BIC (Critério Bayesiano):** $BIC = n \cdot \ln\left(\frac{RSS}{n}\right) + k \cdot \ln(n)$
* **AICc (AIC Corrigido):** $AICc = AIC + \frac{2k(k+1)}{n - k - 1}$

Onde:
* $n$ = número de amostras (pontos).
* $k$ = número de coeficientes (parâmetros) do modelo.
* $y_i$ = valor observado; $\hat{y}_i$ = valor previsto pelo modelo; $\bar{y}$ = média dos valores observados.

---