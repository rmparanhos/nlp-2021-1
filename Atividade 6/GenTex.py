import pandas as pd
from Helper import Model
import operator
from Helper.ViterbiTest import ComputeTranMatrixCharacter
from Helper import TextHelper
from Helper import HMMParam
from Helper.ViterbiHelper import *
import numpy as np
import sys
import os.path
import pickle
from multiprocessing import Process

'''
Reads Model Data From Pickle File
'''
def ReadModelData():
    with open(dataSetFolder+ os.path.join('/Models/ModelData.pkl'),'rb') as f: #alterado por causa do mac
        modelData = pickle.load(f)

    return modelData

'''
Formats the output for user
'''
def FormatOutput(words):
    words = [',' if x=='COMMA' else x for x in words]
    words = ['.' if x=='PERIOD' else x for x in words]
    return words

'''
Generates text based on cereated model
'''
def GenerateText():

    modelData = ReadModelData()

    postValues = posterior(modelData.HMM,modelData.ProbValues)
    
    wordPath = []
    for item in postValues:
        hiValue = max(item.items(), key=operator.itemgetter(1))[0]
        wordPath.append(hiValue)


    WordList = modelData.TextHelper.Encoder.inverse_transform(wordPath)
    WordList = FormatOutput(WordList)

    print("\n")

    print("Generated Text from Text Corpus")
    print("\n**************\n")
    print(' '.join(WordList))

    print("\n\n**************\n\n")


'''
Predicts Text based on the entered sequence and model
'''
def PredictText():
    print("Text Prediction")
    modelData = ReadModelData()

    count = len(modelData.TextHelper.Encoder.classes_)

    sequence = input('Enter a sequence of words: ')

    sequence = modelData.TextHelper.TextProcess(sequence)
    observations =[]
    i = 1
    for word in sequence.split():
        try:
            en = modelData.TextHelper.Encoder.transform([word])[0];
            observations.append(en)
        except ValueError:
            observations.append(len(modelData.TextHelper.Encoder.classes_)+i)
            i = i+1

    path,prob = ViterbiAlgo(modelData.HMM.TransMat,modelData.HMM.TransMat.transpose(1,0),observations)

    WordList = modelData.TextHelper.Encoder.inverse_transform(path.astype(int))
    WordList = FormatOutput(WordList)
    print("The best prediction is '{0}'".format(' '.join(WordList)))

    print("\n")

'''
Retrain the Model - From Dataset
'''
def ReTrain(corpus):
    print("Retraining Model")
    textPath = dataSetFolder+ os.path.join(corpus)
    with open(textPath, 'r') as textFile:
        textData=textFile.read().replace('\n', ' ')

    textHelper = TextHelper.TextHelper()
    textHelper.PreProcessText(textData)
    HMMValues = HMMParam.HMM()
    HMMValues.SetParams(textHelper.EncodedTextList)

    #path = GenerateTextPath(HMMValues.TransMat.copy())
    #vTable = viterbi(HMMValues)

    probValues = ProbValues()
    
    probValues.ForwardValues,probValues.ProbForward = ForwardAlgo(HMMValues,textHelper.Encoder.transform(['PERIOD'])[0])
    probValues.BackwardValues = BackwardAlgo(HMMValues,textHelper.Encoder.transform(['PERIOD'])[0])

    pklModel = Model.Model(HMMValues,textHelper,probValues) 

    with open(dataSetFolder+ os.path.join('/Models/ModelData.pkl'),'wb+') as f:
        pickle.dump(pklModel, f)

    print("Model Retraining Completed")



print("Text Generation/Prediction using HMM")

dataSetFolder = os.path.dirname(os.path.realpath(__file__))

while False:
    print("1. Generate Text Sequence")
    print("2. Predict Text")
    print("3. Re-Train Model")
    print("Any other key to exit")

    userChoice = (input('Select one option from above : '))

    if userChoice == "1":
        GenerateText()
    elif userChoice == "2":
        PredictText()
    elif userChoice == "3":
        ReTrain()
    else:
        choice = input('Please Confirm. Press Y to Restart and N to Exit: ')
        if str.lower(choice) == 'n':
            break
        else:
            continue

#print('End')

