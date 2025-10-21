def erro_absoluto(valor_real,valor_aprox):
    """
    Calcula o erro absoluto entre o valor real/definido/sla e o valor aproximado.

    Parâmetros:
        valor_real(float): Valor considerado verdadeiro.
        valor_aprox(float): Valor aproximado

    Retorna:
        float: O erro absoluto, dado por |valor_real - valor_aprox|.
    """
    return abs(valor_real-valor_aprox)

def erro_relativo(valor_real,valor_aprox):
    """
    Calcula o erro relativo entre o valor real e o valor aproximado.

    Parâmetros:
        valor_real(float): Valor considerado verdadeiro.
        valor_aprox(float): Valor aproximado.

    Retorna:
        float: O erro relativo, dado por |valor_real - valor_aprox| / |valor_real|.
    """
    return abs(valor_real-valor_aprox)/abs(valor_real)
a=float(input("Digite um valor: "))
b=float(input("Digite um valor aproximado para o valor anterior: "))
print(f"Erro absoluto: {erro_absoluto(a,b):.6f}")
print(f"Erro relativo: {erro_relativo(a,b):.8f}")
