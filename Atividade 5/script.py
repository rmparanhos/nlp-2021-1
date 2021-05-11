#Selecionar um dataset usado anteriormente para a tarefa de classificação.
#Treinar um classificador baseado em redes neurais (a rede deve ter, ao menos, a camada de embeddings, 
#uma camada intermediária e a camada de saída).
#Comparar o treinamento dos embeddings com dois tipos de embeddings pré-treinados 
#(pode ser uma variação apenas na dimensão, p.ex, word2vec com 100 ou 200 dim).
#Comparar ao menos duas alterações de hiperparâmetros (tipo da ativação, alteração na arquitetura, 
# otimizador, ajuste da taxa de aprendizado, ajustar ou não a taxa de aprendizado, etc).
#Mostrar casos de troca de acerto por classe entre os diferentes modelos.

#Sergio De Melo Barreto Junior9 de mai.
#Aline, só para ver se eu entendi direito:
#- Experimento 1: Embedding Layer + nn Layer + Layer de classificação (e.: sentimento positivo ou negativo)
#- Experimento  2: W2V +  nn Layer + Layer de classificação (e.: sentimento positivo ou negativo)
#- Comparar desempenho dos experimentos
#- Variar hiperparâmetros do experimento1

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

NUM_ROWS = 100000
#diminuindo tamanho para agilizar
train_df = train_df.head(NUM_ROWS)
test_df = test_df.head(int(NUM_ROWS/10))
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


new_df = pd.concat([comp_df[comp_df['label']==0].head(int(NUM_ROWS/10)), 
comp_df[comp_df['label']==1].head(int(NUM_ROWS)), 
comp_df[comp_df['label']==2].head(int(NUM_ROWS))])
new_df.reset_index(drop=True, inplace=True)
print("Label Não Toxico (0): {}".format(new_df[new_df['label']==0].count()['label']))
print("Label Toxico (1): {}".format(new_df[new_df['label']==1].count()['label']))
print("Label Toxico Severo (2): {}".format(new_df[new_df['label']==2].count()['label']))

#gerando csvs para abrir no tensor
treino_df = new_df[new_df['is_test']==0]
treino_df.to_csv(path_or_buf='treino.csv',index=False)

teste_df = new_df[new_df['is_test']==1]
teste_df.to_csv(path_or_buf='teste.csv',index=False)

#tensor
import torch
import torchtext
from torchtext.legacy import data
#from torchtext.datasets import text_classification
#train_dataset, test_dataset = 'corona_NLP_train.csv', 'corona_NLP_test.csv'
#BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
import torch.nn.functional as F
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_prob, hs1, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(embed_dim, hs1) 
        self.fc2 = nn.Linear(hs1, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = embedded.view(embedded.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        preds = self.softmax(self.fc2(x))
        return preds
        #return self.fc(embedded)

text = data.Field(tokenize="spacy")

text = data.Field(tokenize="spacy", include_lengths = True)
target = data.LabelField()
# define train and test sets
train_data = data.TabularDataset(path="treino.csv",
                                 format="csv",
                                 fields=[
                                         ('comment_text', text),
                                         ('label', data.Field())],
                                 skip_header=True)

test_data = data.TabularDataset(path="teste.csv",
                                format="csv",
                                fields=[
                                        ('comment_text', text),
                                        ('label', data.Field())],
                                skip_header=True)
text.build_vocab(train_data, test_data)

VOCAB_SIZE = len(text.vocab)
#NGRAMS = 2
BATCH_SIZE = 8
EMBED_DIM = 32
NUM_CLASS = 5
DROPOUT_PROB = 0.5
HS1 = 128
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, DROPOUT_PROB, HS1, NUM_CLASS).to(device)

def generate_batch(batch):
  label = torch.tensor([int(entry.label[0]) for entry in batch])
  _text = []
  for entry in batch:
      _entry = []
      for t in entry.comment_text:
          _entry.append(text.vocab.stoi[t])
      _text.append(torch.tensor(_entry,dtype=torch.long))
  offsets = [0] + [len(entry) for entry in _text]
  # torch.Tensor.cumsum returns the cumulative sum
  # of elements in the dimension dim.
  # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)
  offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
  _text = torch.cat(_text)
  #print(' BATCH' , _text)
  #print(offsets)
  return _text, offsets, label

from torch.utils.data import DataLoader

def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        # Clear the gradient buffers of the optimized parameters.
        # Otherwise, gradients from the previous batch would be accumulated.
        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            #print(output, cls)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)

import time
from torch.utils.data.dataset import random_split
import torch.optim as optim
N_EPOCHS = 20
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
#optimizer = optim.SparseAdam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(train_data) * 0.95)
print(train_len)
sub_train_, sub_valid_ = random_split(train_data, [train_len, len(train_data) - train_len])
sub_train_

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)
    
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    
    print(f'Epoch: {epoch + 1}, | time in {mins} minutes and {secs} seconds')
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

print('Checking the results of test dataset...')
test_loss, test_acc = test(test_data)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

def ngrams_iterator(token_list, ngrams):
    """Return an iterator that yields the given tokens and their ngrams.

    Arguments:
        token_list: A list of tokens
        ngrams: the number of ngrams.

    Examples:
        >>> token_list = ['here', 'we', 'are']
        >>> list(ngrams_iterator(token_list, 2))
        >>> ['here', 'here we', 'we', 'we are', 'are']
    """

    def _get_ngrams(n):
        return zip(*[token_list[i:] for i in range(n)])

    for x in token_list:
        yield x
    for n in range(2, ngrams + 1):
        for x in _get_ngrams(n):
            yield ' '.join(x)

NGRAMS = 2
label = {0 : "Não Toxico",
         1 : "Toxico",
         2 : "Toxico Severo"}

def predict(_text, model, vocab, ngrams):
    if len(_text) == 0:
        return 0
    with torch.no_grad():
        _text = [vocab.stoi[token] for token in ngrams_iterator(_text,ngrams)]
        output = model(torch.tensor(_text), torch.tensor([0]))
        return output.argmax(1).item()

model = model.to('cpu')
only_mistakes = True
erros = 0
acertos_toxicos = 0
acertos_toxicos_severos = 0
for entry in test_data:
  prediction = predict(entry.comment_text, model, text.vocab, NGRAMS)
  if only_mistakes and int(entry.label[0]) != prediction:
    erros += 1
    print('\n----ERRO-----')
    print('Label real: ', label[int(entry.label[0])], "\nTexto:", " ".join(entry.comment_text), "\nLabel prevista: ",label[prediction])
  elif prediction == 1:
    acertos_toxicos += 1
    print('\n----ACERTO-TOXICO----')
    print('Label real: ', label[int(entry.label[0])], "\nTexto:", " ".join(entry.comment_text), "\nLabel prevista: ",label[prediction])
  elif prediction == 2:
    acertos_toxicos_severos += 1
    print('\n----ACERTO-TOXICO-SEVERO---')
    print('Label real: ', label[int(entry.label[0])], "\nTexto:", " ".join(entry.comment_text), "\nLabel prevista: ",label[prediction])

print(erros)
print(acertos_toxicos)  
print(acertos_toxicos_severos)
#predictions = [predict(entry.text, model, text.vocab, NGRAMS) for entry in test_data]
