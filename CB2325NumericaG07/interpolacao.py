import numpy as np

def diff_numerica(lista:zip) -> list:
    '''
    Recebe uma sequencia de pontos da forma: [(0, 0)] (pode se fazer zip() de duas arrays ou listas com os valores x e y dos pontos),
    e retorna a derivada numerica em cada ponto,usando diferença central nos pontos centrais,
    e diferença progressiva/regressiva nas pontas.
    '''
    
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


def function_definer(lista_x,lista_y,exception=None):
    '''
    Essa função recebe duas listas e as vincula, retornando uma função vinculo que ao receber um ponto da lista_x retorna um ponto da lista_x
    '''

    func_dicio = dict()
    for x,y in zip(lista_x,lista_y):
        func_dicio[x] = y 
    
    def func(pont):
        if pont in func_dicio.keys():
            return func_dicio[pont]
        else:
            if exception == None:
                raise Exception('Esse ponto não foi definido na função')
            else:
                return exception

    return func


def duplicate(lista) -> list:
    '''
    Duplica cada elemento da lista e mantem a ordem, necessaria para o calculo por exemplo da interpolação de hermite,
    recebe: [1,2,3,4] e retorna: [1,1,2,2,3,3,4,4]
    '''


    l = []
    for i in lista:
        l.append(i)
        l.append(i)
    return l


def ddfunc(Point_list:list,derivada,func)-> list:
    '''
    Recebe a lista de pontos, uma função que retorna as derivadas em cada ponto, e a função que queremos usar na interpolação,
    e retorna as f[] necessarias para o calculo da intepolação de hermite em ordem,
    por exemplo [f[x_0],f[x_0,x_0],f[x_0,x_0,x_1],...] .
    '''
    subslist1,subslist2 = Point_list.copy(),Point_list.copy()#sublist1 e sublist2 são listas que usarei para guardar quais valores serão subtraidos nos denomidaores
    Point_list = [func(p) for p in Point_list] #aplica na lista de pontos a função e retorna cada valor
    
    def der(P_list): #funciona com uma redução de lista, seja x_i o elemento da nova lista e x1_i o elemento da lista antiga de posição i, x_i = (x1_(i+1) - x1_i)/(sublist[i]-sublist2[i]), da mesma forma que seria calcular a interpolação por tabela,  
        new_list = [] #salva nessa lista
        subslist1.pop(0)
        subslist2.pop()
        for i in range(len(P_list)-1):
            if subslist1[i] == subslist2[i]:
                new_list.append(derivada(subslist1[i]))
            else:
                new_list.append((P_list[i+1] - P_list[i])/(subslist1[i] - subslist2[i]))

        return new_list

    result_list = []
    while len(Point_list) != 1: #vai reduzindo a lista até sobrar apenas um elemento, e guarda apenas o topo na tabela, no caso o primeiro da lista
        result_list.append(Point_list[0]) 
        Point_list = der(Point_list)
    result_list.append(Point_list[0])
    return result_list

if __name__ == '__main__':
    print(diff_numerica.__doc__)
    print(ddfunc.__doc__)
    print(duplicate.__doc__)