import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd

#Criar 3 modelos de linguagem com as cartas do Van Gogh e computar a perplexidade de cada modelo. Os 3 modelos são:
#- um modelo 2-gram
#- um modelo 3-gram
#- um modelo 4-gram

#- Separar 70% dos textos para computar as probabilidades e 30% dos textos para testar. Caso computar a probabilidade fique muito pesado, a quantidade de amostras pode ser reduzida.
#- Pode ser necessário usar log para evitar underflow.
#- Para separar os tokens, usar algum framework de tokenização.

def gera_ngram(texto, n):
    ngram_list = []
    for i in range(0,len(texto)-n+1):
        ngram = ""
        for j in range(n):
            ngram = ngram + texto[j+i] + " "        
        ngram_list.append(ngram)
    return ngram_list   

def perplexidade(texto, matriz_probs, n):
    print(texto)
    print(n)
    texto = gera_ngram(texto,2)
    print(texto)
    for ngram in texto:
        tokens = ngram.split()
        col = 0
        row = 0
        for palavra in matriz_probs[0]:
            if palavra == tokens[1]:
                break
            col += 1
        for linha in matriz_probs:
            if linha[0] == tokens[0]:
                break
            row += 1 
        print(tokens)
        print(row,col)
        print(matriz_probs[row][col])
                     

def modela_ngram(diretorio, n_arq_treino, n_arq_teste):
    texto = " "
    for i in range(0, n_arq_treino):
        f = open(diretorio + str(i) + ".txt", "r", encoding="utf-8")
        texto = texto + f.read() + " "
    texto_tokenizado = ["<s>"] #token de inicio
    texto_tokenizado.extend(word_tokenize(texto))
    texto_tokenizado.append("</s>") #token de fim
    nltk.download('punkt')  
    print("Montando 2gram")
    #2gram
    dois_gram = gera_ngram(texto_tokenizado,2)
    c_dois_gram = Counter(dois_gram)
    c_texto_tokenizado = Counter(texto_tokenizado)
    palavras = [""]
    for item in c_texto_tokenizado:
        palavras.append(item)     
    print("Montando matrizes")
    matriz = [palavras]
    matriz_probabilidade = [palavras]
    for a in matriz[0]:
        if a != "":
            linha = [a]
            linha_probabilidade = [a]
            for b in matriz[0]:
                if b != "":
                    # + 1 é o laplace smoothing
                    linha.append(c_dois_gram[(a + " " + b + " ")]+1)
                    linha_probabilidade.append((c_dois_gram[(a + " " + b + " ")]+1)/(len(dois_gram)*2))
            matriz.append(linha)   
            matriz_probabilidade.append(linha_probabilidade) 
    print("Escrevendo resultados em arquivos")
    df_ocorrencias = pd.DataFrame(matriz)
    df_ocorrencias.to_csv(path_or_buf=diretorio + "dois_gram.csv",index=False)
    df_probabilidade = pd.DataFrame(matriz_probabilidade)
    df_probabilidade.to_csv(path_or_buf=diretorio + "dois_gram_probabilidade.csv",index=False)  
    #probs sem laplace
    f = open(diretorio + "dois_gram_ordenado.txt", "w", encoding="utf-8")
    for item in c_dois_gram.most_common():
        f.write(str(item[0]) + " " + str(c_dois_gram[item[0]]) + " " + str(c_dois_gram[item[0]]/len(dois_gram)) + '\n')
    print("2gram finalizado")
    #3gram
    print("Montando 3gram")
    dois_gram = gera_ngram(texto_tokenizado,2)
    tres_gram = gera_ngram(texto_tokenizado,3)
    c_dois_gram = Counter(dois_gram)
    c_tres_gram = Counter(tres_gram)
    c_texto_tokenizado = Counter(texto_tokenizado)
    dois_grams = [""]
    palavras = [""]
    for item in c_dois_gram:
        dois_grams.append(item)
    for item in c_texto_tokenizado:
        palavras.append(item)    
    print("montando matrizes")
    matriz_tres = [dois_grams]
    matriz_probabilidade_tres = [dois_grams]
    #como é tres gram, a ordem importa, primeiro passar por todas as palvras e ir pegando os grams
    for a in palavras:
        if a != "":
            linha = [a]
            linha_probabilidade = [a]
            for b in dois_grams:
                if b != "":
                    # + 1 é o laplace smoothing
                    linha.append(c_tres_gram[(a + " " + b)]+1)
                    linha_probabilidade.append((c_tres_gram[(a + " " + b)]+1)/(len(tres_gram)*2))
            matriz_tres.append(linha)   
            matriz_probabilidade_tres.append(linha_probabilidade) 
    print("escrevendo resultados")
    df_ocorrencias_tres = pd.DataFrame(matriz_tres)
    df_ocorrencias_tres.to_csv(path_or_buf=diretorio + "tres_gram.csv",index=False)
    df_probabilidade_tres = pd.DataFrame(matriz_probabilidade_tres)
    df_probabilidade_tres.to_csv(path_or_buf=diretorio + "tres_gram_probabilidade.csv",index=False)
    #probs sem laplace
    f = open(diretorio + "tres_gram_ordenado.txt", "w", encoding="utf-8")
    for item in c_tres_gram.most_common():
        f.write(str(item[0]) + " " + str(c_tres_gram[item[0]]) + " " + str(c_tres_gram[item[0]]/len(dois_gram)) + '\n')
    print("3gram finalizado")
    #4gram
    print("Montando 4gram")
    tres_gram = gera_ngram(texto_tokenizado,3)
    quatro_gram = gera_ngram(texto_tokenizado,4)
    c_tres_gram = Counter(tres_gram)
    c_quatro_gram = Counter(quatro_gram)
    c_texto_tokenizado = Counter(texto_tokenizado)
    tres_grams = [""]
    palavras = [""]
    for item in c_tres_gram:
        tres_grams.append(item)
    for item in c_texto_tokenizado:
        palavras.append(item)    
    print("montando matrizes")
    matriz_quatro = [tres_grams]
    matriz_probabilidade_quatro = [tres_grams]
    #como é quatro gram, a ordem importa, primeiro passar por todas as palvras e 
    #ir pegando os grams
    for a in palavras:
        if a != "":
            linha = [a]
            linha_probabilidade = [a]
            for b in tres_grams:
                if b != "":
                    # + 1 é o laplace smoothing
                    linha.append(c_quatro_gram[(a + " " + b)]+1)
                    linha_probabilidade.append((c_quatro_gram[(a + " " + b)]+1)/(len(quatro_gram)*2))
            matriz_quatro.append(linha)   
            matriz_probabilidade_quatro.append(linha_probabilidade) 
    print("escrevendo resultados")
    df_ocorrencias_quatro = pd.DataFrame(matriz_quatro)
    df_ocorrencias_quatro.to_csv(path_or_buf=diretorio + "quatro_gram.csv",index=False)
    df_probabilidade_quatro = pd.DataFrame(matriz_probabilidade_quatro)
    df_probabilidade_quatro.to_csv(path_or_buf=diretorio + "quatro_gram_probabilidade.csv",index=False)
    #probs sem laplace
    f = open(diretorio + "quatro_gram_ordenado.txt", "w", encoding="utf-8")
    for item in c_quatro_gram.most_common():
        f.write(str(item[0]) + " " + str(c_quatro_gram[item[0]]) + " " + str(c_quatro_gram[item[0]]/len(tres_gram)) + '\n')
    print("4gram finalizado")
    print(perplexidade(["<s>","This", "little","island","</s>"],matriz_probabilidade,2))

gera_ngram(["<s>","Bom","dia","e","boa","páscoa","<s>"],2)
gera_ngram(["<s>","Bom","dia","e","boa","páscoa","<s>"],3)
gera_ngram(["<s>","Bom","dia","e","boa","páscoa","<s>"],4)
modela_ngram('cartas_tratadas/', 7, 3)