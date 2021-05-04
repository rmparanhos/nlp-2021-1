#1- Usando um texto escolhido por você, mas que seja da literatura Brasileira, 
# julgar o acerto de ao menos 10 casos de 5 tags distintas pelo POS tagger do Spacy. 
# Apontar casos de erro, especificando qual a classe erroneamente utilizada e possíveis razões para isso.
#2 - Avaliar a geração de sentenças do gentext, considerando, para ao menos 4 sentenças geradas, 
# quantos grams você avalia como fluentes usando a base original (ou seja, apenas 2-grams fazem sentido, 
# quantos na sentença toda e quais são eles? ou quantos 3-grams fazem sentido, etc). 
# Repetir o processo para um corpus em português.
#(código oriundo de https://github.com/nareshkumar66675/GenTex e implementação confirmada por mim: HMM para geração de texto)
#3 - Avaliar o código e identificar no que ele difere do modelo visto em aula, 
# considerando a formulação do problema como um HMM.
#(EXTRA) Alterar o código gentext para que o modelo tenha a mesma formulação vista em aula. Repetir a avaliação (2) e comparar.

import spacy
from spacy.lang.pt.examples import sentences 

nlp = spacy.load("pt_core_news_sm")

text = "o_cortico.txt" # O Cortico de Aluisio Azevedo

with open(text, 'r') as textFile:
  textData=textFile.read().replace('\n', ' ')

doc = nlp(textData[29:3166]) #primeiro paragrafo
print("Texto a ser taggeado")
print(doc.text)

#cria lista de tags
taggeados = []
for token in doc:
    taggeados.append((token.text, token.pos_))
print("")
print("Palavras com tags")
print(taggeados)

#Julgar PROPN - Substantivos Proprio
print("")
print("10 Substantivos proprios (PROPN):")
count = 1
for item in taggeados:
  if count == 11:
    break
  if item[1] == "PROPN":
    print("{} - {}".format(count, item[0]))
    count += 1 

#Julgar ADJ - adjetivos 
print("")
print("10 Adjetivos (ADJ):")
count = 1
for item in taggeados:
  if count == 11:
    break
  if item[1] == "ADJ":
    print("{} - {}".format(count, item[0]))
    count += 1 


#Julgar VERB - Verbos
print("")
print("10 Verbos (VERB):")
count = 1
for item in taggeados:
  if count == 11:
    break
  if item[1] == "VERB":
    print("{} - {}".format(count, item[0]))
    count += 1 


#Julgar NUM - Numeral 
print("")
print("10 Numerais (NUM):")
count = 1
for item in taggeados:
  if count == 11:
    break
  if item[1] == "NUM":
    print("{} - {}".format(count, item[0]))
    count += 1 


#Julgar PROPN - Substantivos
print("")
print("10 Substantivos (NOUN):")
count = 1
for item in taggeados:
  if count == 11:
    break
  if item[1] == "NOUN":
    print("{} - {}".format(count, item[0]))
    count += 1 

#Avaliar sentencas do GenTex
print("")
print("------------------------")
print("")
print("Entrando no GenTex")
print("")
from GenTex import GenerateText,ReTrain
print("Treinando o GenTex com o corpus original")
print("")
ReTrain('/Dataset/lorem.txt')
print("")
print("Gerando texto")
GenerateText()

#Avaliar sentencas do GenTex Utilizando um corpus em portugues
print("Treinando o GenTex com o corpus em portugues, o cortico de aluisio azevedo")
ReTrain('/o_cortico_reduzido.txt') 
print("Gerando texto")
GenerateText()