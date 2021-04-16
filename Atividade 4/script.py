#- Escolher um dataset usado anteriormente (da atividade (4) 
# ou das cartas de Van Gogh ou algum outro que você ache interessante) 
# e comparar os resultados de um método probabilístico (LDA) com um algébrico (MNF, SVD).
#- Comparar os resultados e tempo de execução do LDA implementado no 
# gensim (visto em aula) com a implementação do 
# pacote TOMOTOPY (https://github.com/bab2min/tomotopy).
#- EXTRA: Sugerir rótulos para ao menos 3 tópicos, no modelo escolhido por você.

import pandas as pd
import spacy
from spacy.lang.en import English
spacy.load('en_core_web_sm')
parser = English()
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
import gensim
from gensim import corpora
import tomotopy as tp
import time

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.text.isspace():
            continue
        elif token.like_url:
            #lda_tokens.append('URL')
            continue
        elif token.text.startswith('@'):
            #lda_tokens.append('SCREEN_NAME')
            continue
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def main():
  #Abrindo dataset
  train_df = pd.read_csv('jigsaw-toxic-comment-classification-challenge/train.csv')
  test_df = pd.read_csv('jigsaw-toxic-comment-classification-challenge/test.csv')

  #PARA AGILIZAR
  train_df = train_df.head(20000)
  test_df = test_df.head(2000)
 
  tempo = time.time()
  train_data = []
  i = 0
  while i < train_df.shape[0]:
    line = train_df.iloc[i]['comment_text']
    tokens = prepare_text_for_lda(line)
    if tokens:
      train_data.append(tokens)
    i += 1
  print("")
  print("Primeiros 5 itens da base de treino:")
  print(train_data[:5])

  test_data = []
  i = 0
  while i < test_df.shape[0]:
    line = test_df.iloc[i]['comment_text']
    tokens = prepare_text_for_lda(line)
    if tokens:
      test_data.append(tokens)
    i += 1
  print("")
  print("Primeiros 5 itens da base de teste:")
  print(test_data[:5])

  all_data = train_data + test_data
  dictionary = corpora.Dictionary(all_data)
  corpus = [dictionary.doc2bow(token, allow_update=True) for token in train_data]
  print("")
  print("Tempo gasto no pre processamento: {}".format(time.time() - tempo))
  tempo = time.time()
  
  NUM_TOPICS = 5
  print("")
  print("Criando {} topicos utilizando o LDA do gensim".format(NUM_TOPICS))
  ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word = dictionary, num_topics=NUM_TOPICS, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
  topics = ldamodel.print_topics(num_words=10)
  print("Topicos: ")
  for topic in topics:
    print("")
    print(topic)

  print("")
  test_bow = [dictionary.doc2bow(token, allow_update=True) for token in test_data]
  print("Encaixando nos topicos o teste: ")
  print(test_data[0])
  print("Resultado: ")
  print(ldamodel.get_document_topics(test_bow[0]))
  print("")
  print("Tempo gasto no LDA gensim: {}".format(time.time() - tempo))
  tempo = time.time()
  
  #tomotopy
  print("")
  print("Criando 5 topicos utilizando o LDA do tomotopy")
  mdl = tp.LDAModel(k=5)
  
  for line in test_data:
    mdl.add_doc(line)

  for i in range(0, 100, 10):
    mdl.train(10)
    #print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

  print("Topicos:")
  for k in range(mdl.k):
    print("")
    print('Top 10 words of topic #{}'.format(k))
    print(mdl.get_topic_words(k, top_n=10))

  #print("")
  doc_inst = mdl.make_doc(test_data[0])
  topic_dist, ll = mdl.infer(doc_inst)
  print("Topic Distribution for Unseen Docs: ", topic_dist)
  print("Log-likelihood of inference: ", ll)
  print("")
  print("Tempo gasto no LDA tomotopy: {}".format(time.time() - tempo))
  tempo = time.time()
  
  #print(mdl.summary())

main()  