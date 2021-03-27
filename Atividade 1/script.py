import re
import itertools
import collections

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
        texto_limpo = re.sub(remocao_regex, r'', m.group(0))

        #e o termo &nbsp; que aparece em algumas cartas (&nbsp;(1[A-z]:[0-9])*)
        remocao_regex_2 = r'\.[0-9]'
        f_tratado.write(re.sub(remocao_regex_2, r'.', texto_limpo))

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

#implementacao feita por mim do bpe, antes de saber que podiamos utilizar o codigo dado em aula
def bpe_raffael(texto, k):
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

#Parte 2:
#Na segunda parte, você deve executar os algoritmos de tokenização BPE e WordPiece vistos na aula, ambos **implementados** por você. 
#Além do script, você deve informar quantos tokens foram encontrados por cada abordagem e 
#apontar exemplos de possíveis diferenças que cada abordagem pode ter encontrado. 
#Separe 70% das cartas para fazer a criação dos dicionários de tokenização e 30% para fazer o teste da sua implementação 
#(os 30% não podem ser usados ao criar o dicionário). Caso o processo fique lento (Wordpiece pode ser lento), 
#você pode escolher um subconjunto das cartas.

#Implementacao do bpe
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range (len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs        

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def bpe(vocab, num_merges):
    print('Iniciando BPE')
    best_list = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        best_list.append(best)
    print('BPE Finalizado')
    return vocab, best_list


def tokeniza_carta_bpe(diretorio,n_arq,k):
    texto = " "
    for i in range(0, n_arq):
        f = open(diretorio + str(i) + ".txt", "r", encoding="utf-8")
        texto = texto + f.read() + " "
    dicionario = {}   
    print("Inciando processo de criacao do dicionario")
    for palavra in texto.split():
        if " ".join(palavra) + ' </w>' in dicionario.keys():
            dicionario[" ".join(palavra) + ' </w>'] = dicionario[" ".join(palavra) + ' </w>'] + 1
        else:
            dicionario[" ".join(palavra) + ' </w>'] = 1
    vocab, best_list = bpe(dicionario, k)
    print('Escrevendo resultado do BPE num arquivo')
    f = open(diretorio + "vocab.txt", "w", encoding="utf-8")
    f.write("Parametros: n_arq = " + str(n_arq) + ", k = " + str(k) + "\n" + str(vocab))
    print('vocab.txt contem o vocabulario')    
    f = open(diretorio + "best_list.txt", "w", encoding="utf-8")
    f.write("Parametros: n_arq = " + str(n_arq) + ", k = " + str(k) + "\n" + str(best_list))
    print('best_list.txt contem a lista de todos os "bests", pares que sao encontrados mais vezes em uma iteracao')    
    
#trata_carta('cartas_van_gogh_segunda.txt')    
#print(BPE("low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new", 30))
#tokeniza_cartas('cartas_tratadas/', 20, 30)
#bpe({'l o w </w>':5, 'l o w e s t </w>':2, 'n e w e r </w>':6,'w i d e r </w>':3, 'n e w </w>':2}, 3)
tokeniza_carta_bpe('cartas_tratadas/', 700, 50)
