import numpy as np

import joblib
import torch

import config
import dataset
import engine
from model import EntityModel
import io



if __name__ == "__main__":

    meta_data = joblib.load("meta.bin")
    enc_pos = meta_data["enc_pos"]
    enc_tag = meta_data["enc_tag"]

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))
    read_file = list(open(config.INPUT_FILE, "r", encoding='utf8'))
    # print("HEHER")

    sentence = ""
    word_idx = []
    device = torch.device('cuda')
    model = EntityModel(num_tag=num_tag, num_pos=num_pos)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)
    with open(config.OUTPUT_FILE, mode = 'w', newline='', encoding='utf8') as write_file:
        counting = 0
        for word in read_file:
            # print(word)
            
            if(word == "\n"):

                counter = 1
                tokenized_sentence = config.TOKENIZER.encode(sentence)
                sentence = sentence.split()
                # print(sentence)
                for w in sentence:
                    length = len(config.TOKENIZER.encode(w)) - 2
                    word_idx.append(counter)
                    counter += length
                
                # RESET COUNTER
                counter = 1
                

            
                # tokenized_sentence = config.TOKENIZER.encode(sentence)


                #encoded_shit = config.TOKENIZER.tokenize(sentence)

                #print(encoded_shit)

                #sentence = sentence.split()
                
                
                # print(tokenized_sentence)
                # for i in range(len(encoded_shit)):
                #     word = encoded_shit[i]
                #     # print(word)
                #     if len(word) < 2 or (not (word[0] == "#" and word[1] == "#")):
                #         word_indexes.append(i + 1)
                # print(word_indexes)

                test_dataset = dataset.EntityDataset(
                    texts=[sentence], 
                    pos=[[0] * len(sentence)], 
                    tags=[[0] * len(sentence)]
                )

                

                with torch.no_grad():
                    data = test_dataset[0]
                    for k, v in data.items():
                        data[k] = v.to(device).unsqueeze(0)
                    tag, pos, _ = model(**data)

                    tags = (
                        enc_tag.inverse_transform(
                            tag.argmax(2).cpu().numpy().reshape(-1)
                        )[:len(tokenized_sentence)]
                    )
                    pos = (enc_pos.inverse_transform(
                                pos.argmax(2).cpu().numpy().reshape(-1)
                            )[:len(tokenized_sentence)])
                    # print(tags)
                    # print(word_idx)
                    prev_char = ''
                    for y in range(len(word_idx)):
                        x = word_idx[y]
                        next_x = len(tags) - 1 if y == len(word_idx) - 1 else word_idx[y + 1]

                        temp_char = str(tags[x])
                        for i in range(x, next_x):
                            if pos[i] == "I" or pos[i] == "B":
                                temp_char = 'B'

                        # if 'NNP' in str(pos[x]) and temp_char == 'O':
                        #     temp_char == "I"
                        

                        if temp_char == 'I' and (prev_char != 'B' and prev_char != 'I'):
                            temp_char = 'B'

                        if temp_char == 'B' and (prev_char == 'B'):
                            temp_char = 'I'
                        
                        write_file.write(temp_char)
                        write_file.write("\n")
                        prev_char = temp_char
                    write_file.write("\n")
                    #print("HI")
                    if counting < 50:
                        print(counting)
                        print(sentence)
                        print(
                            tags
                        )
                        print(
                            pos
                        )
                    counting += 1
                sentence = ""
                word_idx = []
                # break
            else:
                sentence += str(word + " ")

