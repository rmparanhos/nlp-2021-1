#Escolher um dos datasets abaixo, associados ao problema de detecção de 
#discurso ofensivo ou de discurso de ódio:
#https://github.com/LaCAfe/Dataset-Hatespeech
#https://github.com/punyajoy/Hateminers-EVALITA
#https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data ESCOLHIDO
#https://github.com/UCSM-DUE/IWG_hatespeech_public
#https://figshare.com/articles/Wikipedia_Detox_Data/4054689
#https://github.com/zeerakw/hatespeech
#https://www.kaggle.com/c/detecting-insults-in-social-commentary/
#http://inf.ufrgs.br/~rppelle/hatedetector/
#https://github.com/JAugusto97/ToLD-Br

#Pré-processar o dataset e comparar ao menos três estratégias de pré-processamento 
# (lematizar ou não, fazer stemming ou não, remover stop-words ou não;  
# caso o dataset seja de tweets, remover hashtags ou não, etc).
#Usar BOW + regressão logística para classificação. Exibir as 
# métricas de classificação e a matriz de confusão.
#Exibir estatísticas do dataset: número de exemplos 
# por classe, palavras mais usadas por classe (após remoção de stop-words), etc.
  
#(EXTRA:) Selecionar uma amostra de exemplos classificados incorretamente e 
#tentar elaborar possíveis razões para a classificação incorreta.

import pandas as pd
import spacy
import string
import sklearn 
import seaborn as sns
import matplotlib.pyplot as plt
import collections as cl
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score

#Abrindo dataset
train_df = pd.read_csv('jigsaw-toxic-comment-classification-challenge/train.csv')
test_df = pd.read_csv('jigsaw-toxic-comment-classification-challenge/test.csv')
test_labels_df = pd.read_csv('jigsaw-toxic-comment-classification-challenge/test_labels.csv')

#diminuindo tamanho para agilizar
train_df = train_df.head(5000)
test_df = test_df.head(500)
#concatena pra ter o resultado do teste, depois remove coluna duplicada
test_df = pd.concat([test_df, test_labels_df], axis=1, join="inner").T.drop_duplicates().T

#marca teste e treino
train_df['is_test'] = 0
test_df['is_test'] = 1

# combinar para preproc
comp_df = pd.concat([train_df, test_df])
comp_df.reset_index(drop=True, inplace=True)

#remove as linhas com -1 pois de acordo com o material, -1 nao tem resposta
comp_df = comp_df[comp_df.toxic != -1]

#funcao para criar coluna label a partir das colunas toxic e severe toxic
def label_toxicity (row):
    if row['toxic'] == 1 :
      if row['severe_toxic'] == 1 :
        return 2
      return 1
    if row['severe_toxic'] == 1 :  
      return 2
    return 0

# classificar o grau de toxicicidade, 0 nao toxico, 1 toxico, 2 toxico severe=o
comp_df['label'] = comp_df.apply (lambda row: label_toxicity(row), axis=1)
#reduzindo as colunas
comp_df = comp_df[['comment_text','label','is_test']]
#metrica, numero de exemplos por classe:
print("Label Não Toxico (0): {}".format(comp_df[comp_df['label']==0].count()['label']))
print("Label Toxico (1): {}".format(comp_df[comp_df['label']==1].count()['label']))
print("Label Toxico Severo (2): {}".format(comp_df[comp_df['label']==2].count()['label']))
#Remove URL
comp_df.comment_text = comp_df.comment_text.str.replace(r"http\S+", "")
#Remove \n
comp_df.comment_text = comp_df.comment_text.str.replace(r'\n'," ")
#Remove all non-character
comp_df.comment_text = comp_df.comment_text.str.replace(r"[^a-zA-Z ]","")
# Remove extra space
comp_df.comment_text = comp_df.comment_text.str.replace(r'( +)'," ")
comp_df.comment_text = comp_df.comment_text.str.strip()

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
#OSError: [E941] Can't find model 'en'. It looks like you're trying to load a model from a shortcut, which is deprecated as of spaCy v3.0. To load the model, use its full name instead:
nlp = spacy.load("en_core_web_sm")
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = nlp(sentence)
    # Lemmatizing each token and converting each token into lowercase
    # comentar esta linha para nao utilizar lemmatizacao
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    
    # Removing stop words
    # comentar esta linha para nao remover stop_words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    
    # return preprocessed list of tokens
    return mytokens

comp_df['corpus'] = [spacy_tokenizer(comment_text) for comment_text in comp_df.comment_text]

#metrica, palavras mais usadas na classe 0 ou nao toxico
count_nao_toxico = []
for index, row in comp_df[comp_df['label'] == 0].iterrows():
    count_nao_toxico.extend(row['corpus'])
print("Top 10 palavras nao toxicas por ocorrencias")
print(cl.Counter(count_nao_toxico).most_common(10))
count_toxico = []
for index, row in comp_df[comp_df['label'] == 1].iterrows():
    count_toxico.extend(row['corpus'])
print("Top 10 palavras toxicas por ocorrencias")
print(cl.Counter(count_toxico).most_common(10))
count_toxico_severo = []
for index, row in comp_df[comp_df['label'] == 2].iterrows():
    count_toxico_severo.extend(row['corpus'])
print("Top 10 palavras toxicas severas por ocorrencias")
print(cl.Counter(count_toxico_severo).most_common(10))

comp_df.corpus = comp_df.apply(lambda x: " ".join(x.corpus),axis=1)

x_train=comp_df.corpus[comp_df.is_test==0]
y_train=comp_df.label[comp_df.is_test==0]
x_test=comp_df.corpus[comp_df.is_test==1]
y_test=comp_df.label[comp_df.is_test==1]


# convertendo em BOW com valoração de frequência
freq_vector = CountVectorizer(min_df=5, ngram_range=(1,2),stop_words='english').fit(comp_df.corpus)
x_train = freq_vector.transform(x_train)
x_test = freq_vector.transform(x_test)

# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=500)
# model generation
y_train = y_train.astype(int)
classifier.fit(x_train,y_train)
from sklearn.metrics import precision_recall_fscore_support
y_pred_train=classifier.predict(x_train)
precision, recall, fscore, support = precision_recall_fscore_support(y_train, y_pred_train, average='macro')
print("")
print("Prevendo a base de treino")
print("precisão: {}".format(precision))
print("recall: {}".format(recall))
print("fscore: {}".format(fscore))
print("suport: {}".format(support))
y_pred=classifier.predict(x_test)
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
print("Prevendo a base de teste")
print("precisão: {}".format(precision))
print("recall: {}".format(recall))
print("fscore: {}".format(fscore))
print("suport: {}".format(support))
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

def plot_cm(conf_matrix):
  sns.set(font_scale=1.4,color_codes=True,palette="deep")
  sns.heatmap(cm,annot=True,annot_kws={"size":16},fmt="d",cmap="YlGnBu")
  plt.title("Confusion Matrix")
  plt.xlabel("Predicted Value")
  plt.ylabel("True Value")
  plt.savefig('plot.png')
  #plt.show()

plot_cm(cm)

cvs = cross_val_score(LogisticRegression(random_state=42), x_train, y_train, cv=10, verbose=1, n_jobs=-1).mean()
print("")
print("Utilizando 10-fold cross validation o score foi:")
print(cvs)