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

def perplexidade(texto, ngram):
    print("TODO")

def modela_ngram(diretorio, n_arq_treino, n_arq_teste):
    texto = " "
    for i in range(0, n_arq_treino):
        f = open(diretorio + str(i) + ".txt", "r", encoding="utf-8")
        texto = texto + f.read() + " "
    texto_tokenizado = word_tokenize(texto)
    nltk.download('punkt')
    dois_gram = gera_ngram(texto_tokenizado,2)
    c_dois_gram = Counter(dois_gram)
    c_texto_tokenizado = Counter(texto_tokenizado)
    f = open(diretorio + "ngram.txt", "w", encoding="utf-8")
    palavras = [""]
    for item in c_texto_tokenizado:
        palavras.append(item)
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
    df_ocorrencias = pd.DataFrame(matriz)
    df_ocorrencias.to_csv(path_or_buf=diretorio + "ngram.csv",index=False)
    df_probabilidade = pd.DataFrame(matriz_probabilidade)
    df_probabilidade.to_csv(path_or_buf=diretorio + "ngram_probabilidade.csv",index=False)
    for item in c_dois_gram.most_common():
        f.write(str(item[0]) + " " + str(c_dois_gram[item[0]]) + " " + str(c_dois_gram[item[0]]/len(dois_gram)) + '\n')
    
    #print(gera_ngram(word_tokenize(texto),2))
    #print(gera_ngram(word_tokenize(texto),3))
    #print(gera_ngram(word_tokenize(texto),4))

gera_ngram(["<s>","Bom","dia","e","boa","páscoa","<s>"],2)
gera_ngram(["<s>","Bom","dia","e","boa","páscoa","<s>"],3)
gera_ngram(["<s>","Bom","dia","e","boa","páscoa","<s>"],4)
modela_ngram('cartas_tratadas/', 10, 3)