#!/usr/bin/python

#####
# Description: Functions for reading config and data files
# Author: Heike Adel
# Date: 2016
#####

import gzip

def padAndReduceSentenceToContextsize(sentence, contextsize):
  sentenceList = sentence.split()
  for i in range(4):
    sentenceList.insert(0, "PADDING")
    sentenceList.append("PADDING")
  while len(sentenceList) > contextsize:
    sentenceList.pop(len(sentenceList) - 2) # do not pop PADDING
    if len(sentenceList) > contextsize:
      sentenceList.pop(1) # do not pop PADDING
  while len(sentenceList) < contextsize:
    sentenceList.append("<empty>")
  return sentenceList

def openTokenizedFile(filename, contextsize):
  f = open(filename, 'r')
  labels = []
  texts = []
  for line in f:
    line = line.strip()
    label, sentence = line.split(' :: ')
    labelInt = int(label)
    sentenceReduced = padAndReduceSentenceToContextsize(sentence, contextsize)
    labels.append(labelInt)
    texts.append(sentenceReduced)
  return texts, labels

def openTokenizedFileWithoutPadding(filename, contextsize):
  f = open(filename, 'r')
  labels = []
  texts = []
  for line in f:
    line = line.strip()
    label, sentence = line.split(' :: ')
    labelInt = int(label)
    sentenceReduced = reduceSentenceToContextsize(sentence, contextsize)
    labels.append(labelInt)
    texts.append(sentenceReduced)
  return texts, labels
  
def readConfig(configfile):
  config = {}
  # read config file
  f = open(configfile, 'r')
  for line in f:
    if "#" == line[0]:
      continue # skip commentars
    line = line.strip()
    parts = line.split('=')
    name = parts[0]
    value = parts[1]
    config[name] = value
  f.close()
  return config

def readWordvectors(wordvectorfile):
  wordvectors = {}
  vectorsize = 0
  if ".gz" in wordvectorfile:
    f = gzip.open(wordvectorfile, 'r')
  else:
    f = open(wordvectorfile, 'r')
  count = 0
  for line in f:
    if count == 0:
      count += 1
      continue
    parts = line.split()
    word = parts[0]
    parts.pop(0)
    wordvectors[word] = parts
    vectorsize = len(parts)
  f.close()
  return [wordvectors, vectorsize]

