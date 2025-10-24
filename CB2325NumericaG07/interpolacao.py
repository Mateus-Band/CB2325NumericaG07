print('test')


#vou precisar pra interpolação de hermite
import numpy as np

def diff_numerica(lista:zip) -> list:
    ''' Recebe uma sequencia de pontos da forma: [(0, 0)] (pode se fazer zip() de duas arrays ou listas com os valores x e y dos pontos),
    e retorna a derivada numerica em cada ponto,usando diferença central nos pontos centrais,
    e diferença progressiva/regressiva nas pontas. '''
    lista = list(lista)

    diff_list = lista.copy()

    for index,num in enumerate(lista):
        x,y = num
        if index in [0,len(lista)-1]: #se for um dos pontos das pontas
            if index == 0:
                next_x , next_y = lista[index + 1]
                diff_list[index] = (next_y - y)/(next_x - x)  
            
            else:
                prev_x,prev_y = lista[index - 1]
                diff_list[index] = (y - prev_y)/(x - prev_x)

        else:
            next_x , next_y = lista[index + 1]
            prev_x,prev_y = lista[index - 1]
            diff_list[index] = (next_y - prev_y)/(next_x - prev_x)
    
    return diff_list