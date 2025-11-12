Adicionar documentação
# `intepolacao.py` - Módulo de Integração Numérica

Esse modulo fornece ferramentas para a interpolação de pontos.

## Dependências

Este módulo requer as seguintes bibliotecas Python para sua funcionalidade completa:

* **NumPy**: Para operações numéricas e arrays.
* **SymPy**: Para manipulação de expressões simbólicas.
* **Matplotlib**: Para a visualização gráfica (opcional).


## Funções

## `diff_numerica`

Calcula as derivadas numéricas de uma sequencia de pontos.

#### Parâmetros

* **`x`** (`list`): lista das coordenadas x dos pontos.
* **`y`** (`list`): lista das coordenadas y dos pontos.

#### Retorna

* **`list`**: lista das derivadas em cada um dos pontos.


## `_function_definer`

Retorna uma função que relaciona os pontos de uma lista x com uma lista y.

#### Parâmetros

* **`Lista_x`** (`list`) : Lista x que será o dominio da função.
* **`Lista_x`** (`list`) : Lista y que será a imagem da função.
* **`exception`** : Valor, ou qualquer outra coisa que a função retornará se aplicada a um valor fora do dominio (Lista x).

#### Retorna

* **`function`**: Função que relaciona os valores do dominio com os respectivos valores da imagem.



## `_duplicate`

Retorna uma lista com os valores duplicados da forma [1,2,3] -> [1,1,2,2,3,3].

#### Parâmetros

* **`Lista`** (`list`) : Lista x que será duplicada.

#### Retorna

* **`list`**: Lista duplicada.


## `__hermite_ddfunc`

Calcula os valores das diferenças divididas de uma função considerando pontos duplicados como derivada.

#### Parâmetros

* **`Point_list`** (`list`) : Lista das coordenadas x dos pontos que serão interpolados,(Funciona para lista de pontos normais, mas para a hermite mas especificamente é necessário que a lista seja duplicada da forma [1,2,3] -> [1,1,2,2,3,3]).
* **`derivada`** (`function`) : Função que dado um dos valores das coordenadas x dos pontos de Point_list, retorne a derivada no ponto.
* **`func`** (`function`): Função que dado um valor das coordenadas x, retorna o respectivo valor da coordenada y.

#### Retorna

* **`list`**: Retorna uma lista com o os coeficientes necessários para a interpolação de hermite em ordem de uso .


## `interpolacao_de_hermite`

Calcula a função da interpolação de hermite.

#### Parâmetros

* **`x`** (`list`) : Lista das coordenadas x dos pontos escolhidos para a interpolação.
* **`y`** (`list`) : Lista das coordenadas y dos pontos escolhidos para a interpolação.
* **`plot`** (`bool`): Determina se o plot deve acontecer ou não.
* **`grid`** (`bool`): Determina se o plot deve ter grid ou não.

#### Retorna

* **`function`**: Retorna uma função que calcula os valores para a função interpolada.

#### Matemática

A interpolação de hermite funciona com as diferenças divididas da forma:

$$
H(x) = f[x_0] + f[x_0,x_0](x - x_0) + f[x_0,x_0,x_1](x - x_0)^2 + f[x_0,x_0,x_1,x_1](x - x_0)^2 (x - x_1) ...
$$




## `_newton_ddfunc`

Calcula os valores das diferenças divididas de uma função sem valores repetidos.

#### Parâmetros

* **`Point_list`** (`list`) : Lista das coordenadas x dos pontos que serão interpolados.
* **`func`** (`function`): Função que dado um valor das coordenadas x, retorna o respectivo valor da coordenada y.

#### Retorna

* **`list`**: Retorna uma lista com o os coeficientes necessários para a interpolação de newton em ordem de uso.



## `interpolacao_de_newton`


Calcula a função da interpolação de hermite.

#### Parâmetros

* **`x`** (`list`) : Lista das coordenadas x dos pontos escolhidos para a interpolação.
* **`y`** (`list`) : Lista das coordenadas y dos pontos escolhidos para a interpolação.
* **`plot`** (`bool`): Determina se o plot deve acontecer ou não.
* **`grid`** (`bool`): Determina se o plot deve ter grid ou não.

#### Retorna

* **`function`**: Retorna uma função que calcula os valores para a função interpolada.

#### Matemática

A interpolação de newton funciona com as diferenças divididas da forma:

$$
H(x) = f[x_0] + f[x_0,x_1](x - x_0) + f[x_0,x_1,x_2](x - x_0)(x - x_1) + ...
$$


