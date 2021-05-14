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
#- Variar hiperparâmetros do experimentos

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torchtext.legacy import data
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split
from sklearn.metrics import confusion_matrix


#Abrindo dataset
train_df = pd.read_csv('jigsaw-toxic-comment-classification-challenge/train.csv')
test_df = pd.read_csv('jigsaw-toxic-comment-classification-challenge/test.csv')
test_labels_df = pd.read_csv('jigsaw-toxic-comment-classification-challenge/test_labels.csv')

NUM_ROWS = 5000
LR = 1
#diminuindo tamanho para agilizar
train_df = train_df.head(NUM_ROWS)
test_df = test_df.head(int(NUM_ROWS/5))
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
print("Dataset original")
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


new_df = pd.concat([comp_df[(comp_df['label']==0) & (comp_df['is_test']==0)].head(int(NUM_ROWS/7)), 
comp_df[(comp_df['label']==0) & (comp_df['is_test']==1)],
comp_df[comp_df['label']==1].head(int(NUM_ROWS)), 
comp_df[comp_df['label']==2].head(int(NUM_ROWS))])
new_df.reset_index(drop=True, inplace=True)
#print("Label Não Toxico (0): {}".format(new_df[new_df['label']==0].count()['label']))
#print("Label Toxico (1): {}".format(new_df[new_df['label']==1].count()['label']))
#print("Label Toxico Severo (2): {}".format(new_df[new_df['label']==2].count()['label']))

#gerando csvs para abrir no tensor
print("")
print("Após divisão e rebalanceamento de labels")
print("")
treino_df = new_df[new_df['is_test']==0]
print("No treino:")
print("Label Não Toxico (0): {}".format(treino_df[treino_df['label']==0].count()['label']))
print("Label Toxico (1): {}".format(treino_df[treino_df['label']==1].count()['label']))
print("Label Toxico Severo (2): {}".format(treino_df[treino_df['label']==2].count()['label']))
print("")
treino_df.to_csv(path_or_buf='treino.csv',index=False)


teste_df = new_df[new_df['is_test']==1]
print("No teste:")
print("Label Não Toxico (0): {}".format(teste_df[teste_df['label']==0].count()['label']))
print("Label Toxico (1): {}".format(teste_df[teste_df['label']==1].count()['label']))
print("Label Toxico Severo (2): {}".format(teste_df[teste_df['label']==2].count()['label']))
print("")
teste_df.to_csv(path_or_buf='teste.csv',index=False)

#tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_prob, hs1, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.relu = nn.ReLU()
        #self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(embed_dim, hs1) 
        self.fc2 = nn.Linear(hs1, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        #self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = embedded.view(embedded.shape[0], -1)
        x = self.relu(self.fc1(x))
        #x = self.tanh(self.fc1(x))
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
#text.build_vocab(train_data, test_data)

#VOCAB_SIZE = len(text.vocab)
##NGRAMS = 2
#BATCH_SIZE = 8
#EMBED_DIM = 32
#NUM_CLASS = 5
#DROPOUT_PROB = 0.5
#HS1 = 128
#model = TextSentiment(VOCAB_SIZE, EMBED_DIM, DROPOUT_PROB, HS1, NUM_CLASS).to(device)

text.build_vocab(train_data, test_data)		#text.build_vocab(train_data, test_data)
MAX_VOCAB_SIZE = 25_000
text.build_vocab(train_data, test_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 #vectors = "glove.6B.100d", 
                 #vectors = "glove.twitter.27B.100d",
                 vectors = "fasttext.en.100d",
                 unk_init = torch.Tensor.normal_)
#PAD_IDX = text.vocab.stoi[text.pad_token]
VOCAB_SIZE = len(text.vocab)
NGRAMS = 2
BATCH_SIZE = 8
EMBED_DIM = 32
NUM_CLASS = 5
DROPOUT_PROB = 0.5
HS1 = 128
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, DROPOUT_PROB, HS1, NUM_CLASS, True).to(device)
pretrained_embeddings = text.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = text.vocab.stoi[text.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBED_DIM)
#model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBED_DIM)
#print(model.embedding.weight.data)

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


N_EPOCHS = 20
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
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
erros = 0
acertos = 0

previsao_nao_toxicos = 0
previsao_toxicos = 0
previsao_toxicos_severos = 0

acertos_nao_toxicos = 0
acertos_toxicos = 0
acertos_toxicos_severos = 0

erros_nao_toxicos = 0
erros_toxicos = 0
erros_toxicos_severos = 0

#para a matriz de confusao:
original = []
previsoes = []

for entry in test_data:
    prediction = predict(entry.comment_text, model, text.vocab, NGRAMS)
    if prediction == int(entry.label[0]):
        acertos += 1
        if prediction == 0:
            previsao_nao_toxicos += 1 
            acertos_nao_toxicos += 1
        elif prediction == 1:
            previsao_toxicos += 1
            acertos_toxicos += 1
        elif prediction == 2:
            previsao_toxicos_severos += 1
            acertos_toxicos_severos += 1
    elif prediction != int(entry.label[0]):
        erros += 1
        if prediction == 0:
            previsao_nao_toxicos +=1
            erros_nao_toxicos += 1
        elif prediction == 1:
            previsao_toxicos +=1
            erros_toxicos += 1
        elif prediction == 2:
            previsao_toxicos_severos += 1
            erros_toxicos_severos += 1
        print('\n----ERRO---')
        print('Label real: ', label[int(entry.label[0])], "\nTexto:", " ".join(entry.comment_text), "\nLabel prevista: ",label[prediction])
    
    original.append(int(entry.label[0]))
    previsoes.append(prediction)

print("")
print("Erros: ",erros)
print("Acertos: ",acertos)

print("Previsoes nao toxicos: ",previsao_nao_toxicos)
print("Acertos nao toxicos: ",acertos_nao_toxicos)
print("Erros nao toxicos: ",erros_nao_toxicos)

print("Previsoes toxicos: ",previsao_toxicos)
print("Acertos toxicos: ",acertos_toxicos)
print("Erros toxicos: ",erros_toxicos)

print("Previsoes toxicos severos: ",previsao_toxicos_severos)
print("Acertos toxicos severos: ",acertos_toxicos_severos)
print("Erros toxicos severos: ",erros_toxicos_severos)

cm=confusion_matrix(original, previsoes)

def plot_cm(conf_matrix):
  sns.set(font_scale=1.4,color_codes=True,palette="deep")
  sns.heatmap(cm,annot=True,annot_kws={"size":16},fmt="d",cmap="YlGnBu")
  plt.title("Confusion Matrix")
  plt.xlabel("Predicted Value")
  plt.ylabel("True Value")
  plt.savefig('plot.png')
plot_cm(cm)
#predictions = [predict(entry.text, model, text.vocab, NGRAMS) for entry in test_data]
