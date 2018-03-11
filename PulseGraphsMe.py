import os
import numpy as np
import matplotlib.pyplot as plt


def cos(x):
    return np.cos(180 * x / np.pi)

def sin(x):
    return np.sin(180 * x / np.pi)

def inverse(x):
    return 1/x

# Ozii Function
def transformer(x):
    y = cos(x)
    y = sin(y)
    y = inverse(y)
    return y

# X
x = np.linspace(0, 1, 1001)
x = x[1:]

# Return y for a single word
def transform(text,d=0):
    n = len(text)
    y = 0+d
    for i in range(len(text)):
        y += np.log(ord(text[i])/4.) * (x ** (i+1))
    y = transformer(y)
    max_y = np.max(np.abs(y))
    y = (0.5/max_y) * y
    return y

# y for a sentence
def sentence_transformer(sentence,d=0):
    words = sentence.split()
    y = np.zeros(x.shape)
    for i, word in enumerate(words):
        y += transform(word,d)
    max_y = np.max(np.abs(y))
    y = (0.5/max_y) * y
    return y


# ~ sentence = 'Ever since I watched the movie'
# ~ sentence = "Садитесь, пожалуйста. Спасибо, я пешком постою"
# ~ sentence = "哈哈哈"
# ~ sentence = "良好"
# ~ sentence = "хаха"

phrasa = "Стране 44 года. Местных 11 кажется процентов. Из-за логичных для такой ситуации близкородственных связей высок процент детей с инвалидностями но детских домов тут нет. Нищих нет, или ты местный или рабочая виза или турист. Но много женщин пониженной социальной ответственности."

fig = plt.figure(figsize=(15, 15))
x =16* x

n = int(np.sqrt(len(phrasa.split('. '))))+1
print(n)
for i, sentence in enumerate(phrasa.split('.')):
    y1 = sentence_transformer(sentence)
    y2 = sentence_transformer(sentence,0.9)
    
    y1 = 2*y1
    y2 = 2*y2
    with plt.xkcd(scale=4, randomness=5):
        plt.subplot(n,n,i+1, projection='polar')
        #fig = plt.figure(figsize=(10, 10))
        plt.polar(2*np.pi*x, y1, linewidth=2, c='b')
        plt.polar(2*np.pi*x, y2, linewidth=2, c='b')
        plt.fill_between(2*np.pi*x, y1, y2, facecolor='b')
        plt.axis('off')
        #plt.title(sentence, fontsize=18, fontweight='bold')
        fig.tight_layout()


plt.show()
