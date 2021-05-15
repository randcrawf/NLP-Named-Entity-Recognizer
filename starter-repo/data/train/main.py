import io
#mport pandas as pd
#import numpy as np
import csv
import nltk
def process(read_file, write_file):
    #nltk.download()
    f = list(open(read_file, "r"))
    #f = f.split("\n\n")
    sentence_num = 1
    sentence = list()
    with open(write_file, mode = 'w', newline='', encoding='utf-8') as text_file:
        writer = csv.writer(text_file, delimiter=',', quotechar='"')
        writer.writerow(['Sentence #','Word','POS','Tag'])
        print("here")

        for x in f:        
            word = x.split()
            if(len(word) == 0):

                for i, w in enumerate(sentence):
                    if i == 0:
                        writer.writerow([f'Sentence: {sentence_num}',w[0],w[1],w[2]])
                    else:
                        writer.writerow(['',w[0],w[1],w[2]])
                    #
                    # print(word[1])
                    # print(tags[i])
                sentence = list()
                sentence_num += 1
            else:
                sentence.append((nltk.pos_tag(word)[0][0],nltk.pos_tag(word)[0][1], word[1]))

        #print(temp)

        
        
            
        #print(type(x))
        
    # df = pd.read_csv(data_path, encoding="latin-1")
    # df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

    # enc_pos = preprocessing.LabelEncoder()
    # enc_tag = preprocessing.LabelEncoder()

    # df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])
    # df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    # sentences = df.groupby("Sentence #")["Word"].apply(list).values
    # #pos = df.groupby("Sentence #")["POS"].apply(list).values
    # tag = df.groupby("Sentence #")["Tag"].apply(list).values
    # return sentences, pos, tag, enc_pos, enc_tag

def write_to_csv(file):
    with open(file, mode = 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        writer.writerow(['Sentence #','Word','POS','Tag'])
        print("here")


if __name__ == "__main__":
    process('train.txt', '../../ner-solution/train.csv')
    # write_to_csv('../../ner-solution/train.csv')