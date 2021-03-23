import re
import itertools
#Parte 1
#O arquivo com as cartas é um único TXT contém uma marcação ' ","------------------------- 1 =' para indicar o início de cada carta.
#Seu script deve conter: a leitura do arquivo fonte, o tratamento com expressões regulares e a escrita das cartas tratadas. 
#Você pode separar cada carta em uma célula de um arquivo csv ou criar arquivos separados para cada carta. 
#Tal separação deve fazer parte do seu script, de qualquer forma.
#Seu script deve apontar as expressões regulares e descrever (pode ser como comentário) 
#o que você pretende tratar com cada uma.
def trata_carta(arq):
    f = open(arq, "r", encoding="utf-8")
    texto = f.read()
    
    #regex para caputra de cada carta, composto pelo inicio da carta (------------------------- *[0-9]* =), 
    #conteudo da carta (.*?), 
    #inicio do marcador da proxima carta (",")
    carta_regex = r'------------------------- *[0-9]* =.*?","'
    count = 0
    for m in re.finditer(carta_regex, texto, flags=re.DOTALL):
        print('%02d-%02d: %s' % (m.start(), m.end(), m.group(0)))
        f_tratado = open('cartas_tratadas/' + str(count) + ".txt", "w", encoding="utf-8")
        
        #regex para fazer limpeza, caputara os marcadores de inicio de carta (------------------------- *[0-9]* = ), 
        #o termo [sketch X] que aparece em algumas cartas (\[sketch [A-z]\]), 
        #o lixo deixado pelo processo de escrita das cartas (","),
        #e o termo &nbsp; que aparece em algumas cartas (&nbsp;(1[A-z]:[0-9])*)
        remocao_regex = r'------------------------- *[0-9]* = |\[sketch [A-z]\]|","|&nbsp;(1[A-z]:[0-9])*'
        f_tratado.write(re.sub(remocao_regex, r'', m.group(0)))
        count += 1

#Parte 2:
#Na segunda parte, você deve executar os algoritmos de tokenização BPE e WordPiece vistos na aula, ambos **implementados** por você. 
#Além do script, você deve informar quantos tokens foram encontrados por cada abordagem e 
#apontar exemplos de possíveis diferenças que cada abordagem pode ter encontrado. 
#Separe 70% das cartas para fazer a criação dos dicionários de tokenização e 30% para fazer o teste da sua implementação 
#(os 30% não podem ser usados ao criar o dicionário). Caso o processo fique lento (Wordpiece pode ser lento), 
#você pode escolher um subconjunto das cartas.
def tokeniza_cartas(diretorio,n_arq,k):
    texto = " "
    for i in range(0,n_arq):
        f = open(diretorio + str(i) + ".txt", "r", encoding="utf-8")
        texto = texto + f.read() + " "
    print(texto)    
    print(BPE(texto,k))

def BPE(texto, k):
    #processo que cria o dicionario e o vocabulario
    dicionario = {}
    vocabulario = ['_']
    print("processo que cria o dicionario e o vocabulario")
    for palavra in texto.split():
        if " ".join(palavra) + ' _' in dicionario.keys():
            dicionario[" ".join(palavra) + ' _'] = dicionario[" ".join(palavra) + ' _'] + 1
        else:
            dicionario[" ".join(palavra) + ' _'] = 1
            for letra in palavra:
                if letra not in vocabulario:
                    vocabulario.append(letra)
    
    print("encontra as combinacoes")
    for n in range(0,k):
        ocorrencias = {}        
        #gera as combinacoes
        for combinacao in list(itertools.permutations(vocabulario,2)):
            count = 0
            for chave in dicionario.keys():
                if "".join(combinacao) in chave.replace(" ",""):
                    #print("".join(combinacao) + " EM " + chave)
                    count = count + dicionario[chave]
            if count > 0:        
                ocorrencias["".join(combinacao)] = count
            
        #encontra a combinacao com maior ocorrencia
        while(True):
            maior_ocorrencia = []
            for chave in ocorrencias.keys():
                if len(maior_ocorrencia) == 0:
                    maior_ocorrencia = [chave,ocorrencias[chave]]
                if ocorrencias[chave] >= maior_ocorrencia[1]:
                    maior_ocorrencia = [chave,ocorrencias[chave]]
            if maior_ocorrencia[0] in vocabulario:
                ocorrencias[maior_ocorrencia[0]] = 0
            else:
                print(maior_ocorrencia)
                vocabulario.append(maior_ocorrencia[0])
                break
    
    print("montado o vocabularo, é o momento da tokenizacao de fato")
    for token in vocabulario:
        if len(token) > 1:
            for chave in list(dicionario.keys()):
                regex_token = r""
                for char in token:
                    regex_token = regex_token + char + " *"  
                regex_token = regex_token + "?"    
                chave_token = re.sub(regex_token,token,chave)
                #print(regex_token)
                #print(token)
                #print(chave)
                #print(chave_token)
                if chave_token != chave:
                    valor = dicionario[chave]
                    dicionario.pop(chave)
                    dicionario[chave_token] = valor
    return vocabulario, dicionario
    
#trata_carta('cartas_van_gogh_segunda.txt')    
#print(BPE("low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new", 30))
tokeniza_cartas('cartas_tratadas/', 20, 30)