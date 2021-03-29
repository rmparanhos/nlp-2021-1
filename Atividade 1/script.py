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

#Implementacao do max_match/tokenizacao
def max_match_bpe(string, dictionary):
    if len(string) == 0:
        return []
    for i in range(0,len(string)):
        if i == 0:
            firstword = string
            remainder = ''
        else:
            firstword = string[:-i]
            remainder = string[-i:]
        if remainder in dictionary:
            return [max_match_bpe(firstword,dictionary), remainder]

def tokeniza_carta_bpe(diretorio,n_arq_treino,n_arq_teste,k):
    texto = " "
    for i in range(0, n_arq_treino):
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
    print('BPE encontrou ' + str(len(best_list)) + ' tokens')
    f = open(diretorio + "vocab_bpe.txt", "w", encoding="utf-8")
    f.write("Parametros: n_arq_treino = " + str(n_arq_treino) + ", k = " + str(k) + "\n" + str(vocab))
    print('vocab_bpe.txt contem o vocabulario do bpe')    
    f = open(diretorio + "best_list_bpe.txt", "w", encoding="utf-8")
    f.write("Parametros: n_arq_treino = " + str(n_arq_treino) + ", k = " + str(k) + "\n" + str(best_list))
    print('best_list_bpe.txt contem a lista de todos os "bests", pares que sao encontrados mais vezes em uma iteracao')    
    print("Inciando MaxMatch / tokeninzacao")
    #transforma o texto de teste no formato do maxmatch
    texto = " "
    for i in range(n_arq_teste, n_arq_teste*2):
        f = open(diretorio + str(i) + ".txt", "r", encoding="utf-8")
        texto = texto + f.read() + " "
    texto_lista = []    
    for palavra in texto.split():
        texto_lista.append(palavra + '</w>')
    #transforma o resultado do bpe no formato do maxmatch
    best_list_max_match = []
    for item in best_list:
        best_list_max_match.append(item[0]+item[1])

    f = open(diretorio + "tokenizado_bpe.txt", "w", encoding="utf-8")
    f.write("Parametros: algoritmo = MaxMatch, n_arq_teste = " + str(n_arq_teste) + ", dictionary = best_list_bpe.txt" + "\n")    
    f = open(diretorio + "tokenizado_bpe.txt", "a", encoding="utf-8")
    for item in texto_lista:
        f.write(item + ' -- ' + str(max_match_bpe(item,best_list_max_match)) + '\n')
    print('tokenizacao_bpe.txt contem a tokenizacao, utilizando MaxMatch, dos arquivos de testes a partir do bpe montado usando os arquivos de teste')    
    print("MaxMatch / tokeninzacao finalizado")
    
#trata_carta('cartas_van_gogh_segunda.txt')    
tokeniza_carta_bpe('cartas_tratadas/', 600, 310, 30)
#print(max_match('intention',["in", "tent","intent","tent", "tention", "tion", "ion"]))
