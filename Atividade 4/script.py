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
  train_df = train_df.head(1000)
  test_df = test_df.head(100)

  train_data = []
  i = 0
  while i < train_df.shape[0]:
    line = train_df.iloc[i]['comment_text']
    tokens = prepare_text_for_lda(line)
    if tokens:
      train_data.append(tokens)
    i += 1
  print("")
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
  print(test_data[:5])

  all_data = train_data + test_data
  dictionary = corpora.Dictionary(all_data)
  corpus = [dictionary.doc2bow(token, allow_update=True) for token in train_data]
  print("")
  print(dictionary)

  NUM_TOPICS = 5
  ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word = dictionary, num_topics=NUM_TOPICS, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
  topics = ldamodel.print_topics(num_words=10)
  print("")
  for topic in topics:
    print("")
    print(topic)
  print("")

  test_bow = [dictionary.doc2bow(token, allow_update=True) for token in test_data]
  print("")
  print(test_data[0])
  print(ldamodel.get_document_topics(test_bow[0]))


main()  