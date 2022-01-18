#imprting libraries
import pandas as pd
import numpy as np
import json

#reading the training file
df = pd.read_csv("./data/train", sep = "\t", names = ['id', 'words', 'pos'])
#finding the number of occurences for each word
df['occ'] = df.groupby('words')["words"].transform('size')
#function to replace each word with <unk> whose occurrence is less than 2
def replace(row):
    if row.occ < 2:
        return "<unk>"
    else:
        return row.words
#applying the replace function on train data
df['words'] = df.apply(lambda row : replace(row), axis = 1)
print('The Selected Threshold for unkown word replacement is: 2')

#storing unique words in df_vocab and arranging them in descending order of their occurrence value
df_vocab = df.words.value_counts().rename_axis('words').reset_index(name = 'occ')
#finding the data row containing word <unk>
df_unk = df_vocab[df_vocab['words'] == "<unk>"]
#removing the data row with word as <unk> and moving it to the top
index = df_vocab[df_vocab.words == "<unk>"].index
df_vocab = df_vocab.drop(index)
df_vocab = pd.concat([df_unk, df_vocab]).reset_index(drop = True)
#storing the index of each vocabulary in df_vocab
df_vocab['id'] = df_vocab.index + 1
#rearranging the vocabulary dataset into the required format
cols = df_vocab.columns.tolist()
cols = [cols[0], cols[-1], cols[1]]
df_vocab = df_vocab[cols]
#storing the vocabulary dataset to 'vocab.txt'
df_vocab.to_csv("vocab.txt", sep="\t", header=None)
print('The total size of our vocabulary is: {}'.format(df_vocab.shape[0]))
print('The total occurences of the special token \'<unk>\': {}'.format(int(df_vocab[df_vocab["words"] == "<unk>"].occ)))

#storing all the unique tags in training data into 'tags' variable
df_pos = df.pos.value_counts().rename_axis('pos').reset_index(name = 'count')
pos_dict = dict(df_pos.values)
tags = df_pos.pos.tolist()

#storing the training data in a list( called sentences) consisting of lists(called sentence), 
#where each sublist consists of tuples, where each tuple corresponds to one data row in training data
sentences = []
sentence = []
first = 1
for row in df.itertuples():
    if(row.id == 1 and first == 0):
        sentences.append(sentence)
        sentence = []
    first = 0
    sentence.append((row.words, row.pos))
sentences.append(sentence)

#get_trans_matrix function computes the transition matrix for the given sentences
def get_trans_matrix(sentences, tags):
    tr_matrix = np.zeros((len(tags),len(tags)))

    tag_occ = {}
    for tag in range(len(tags)):
        tag_occ[tag] = 0
    
    for sentence in sentences:
        for i in range(len(sentence)):
            tag_occ[tags.index(sentence[i][1])] += 1
            if i == 0: continue
            tr_matrix[tags.index(sentence[i - 1][1])][tags.index(sentence[i][1])] += 1
    
    for i in range(tr_matrix.shape[0]):
        for j in range(tr_matrix.shape[1]):
            if(tr_matrix[i][j] == 0) : tr_matrix[i][j] = 1e-10
            else: tr_matrix[i][j] /= tag_occ[i]

    return tr_matrix

#get_emission_matrix function computes the emission matrix for the given sentences
def get_emission_matrix(tags, vocab, sentences):
    em_matrix = np.zeros((len(tags), len(vocab)))

    tag_occ = {}
    for tag in range(len(tags)):
        tag_occ[tag] = 0

    for sentence in sentences:
        for word, pos in sentence:
            tag_occ[tags.index(pos)] +=1
            em_matrix[tags.index(pos)][vocab.index(word)] += 1

    for i in range(em_matrix.shape[0]):
        for j in range(em_matrix.shape[1]):
            if(em_matrix[i][j] == 0) : em_matrix[i][j] = 1e-10
            else: em_matrix[i][j] /= tag_occ[i]

    return em_matrix

vocab = df_vocab.words.tolist()

#get_trans_probs function coverts the transition matrix to transition dictionary
def get_trans_probs(tags, tr_matrix,prior_prob):
    tags_dict = {}

    for i, tags in enumerate(tags):
        tags_dict[i] = tags

    trans_prob = {}
    for i in range(tr_matrix.shape[0]):
        trans_prob['(' + '<\S>' + ',' + tags_dict[i] + ')'] = prior_prob[tags_dict[i]]
    for i in range(tr_matrix.shape[0]):
        for j in range(tr_matrix.shape[1]):
            trans_prob['(' + tags_dict[i] + ',' + tags_dict[j] + ')'] = tr_matrix[i][j]


    return trans_prob

#get_emission_probs function coverts the emission matrix to emission dictionary
def get_emission_probs(tags, vocab, em_matrix):
    tags_dict = {}

    for i, tags in enumerate(tags):
        tags_dict[i] = tags

    emission_probs = {}

    for i in range(em_matrix.shape[0]):
        for j in range(em_matrix.shape[1]):
            emission_probs['(' + tags_dict[i] + ', ' + vocab[j] + ')'] = em_matrix[i][j]

    return emission_probs

#get_all_prob function generates the transition matrix,emission matrix, 
#transition dictionary and the emission dictionary
def get_all_prob(tags, vocab, sentences,prior_prob):
    tr_matrix = get_trans_matrix(sentences, tags)
    em_matrix = get_emission_matrix(tags, vocab, sentences)
                
    transition_probability = get_trans_probs(tags, tr_matrix,prior_prob)
    emission_probability = get_emission_probs(tags, vocab, em_matrix)

    return transition_probability, emission_probability

#get_inital_prob function calculates the initial transition probability for each tag
def get_inital_prob(df, tags):
    tags_start_occ = {}
    total_start_sum = 0
    for tag in tags:
        tags_start_occ[tag] = 0
    
    for row in df.itertuples():
        if(row[1] == 1):
            tags_start_occ[row[3]]+=1
            total_start_sum += 1
    
    prior_prob = {}
    for key in tags_start_occ:
        prior_prob[key] = tags_start_occ[key] / total_start_sum
    
    return prior_prob

prior_prob = get_inital_prob(df, tags)
trans_prob, em_prob = get_all_prob(tags, vocab, sentences,prior_prob)

print('The number of Transition Parameters are: {}'.format(len(trans_prob)))
print('The number of Emission Parameters are: {}'.format(len(em_prob)))
#we store the transition and emission dictionaries in a json file named 'hmm.json'
with open('hmm.json', 'w') as f:
    json.dump({"transition": trans_prob, "emission": em_prob}, f, ensure_ascii=False, indent = 4)
    
#reading the validation file
validation_data = pd.read_csv("./data/dev", sep = '\t', names = ['id', 'words', 'pos'])
validation_data['occ'] = validation_data.groupby('words')['words'].transform('size')

#storing the validation data in a list( called sentences) consisting of lists(called sentence), 
#where each sublist consists of tuples, where each tuple corresponds to one data row in validation data
valid_sentences = []
sentence = []
first = 1
for row in validation_data.itertuples():
    if(row.id == 1 and first == 0):
        valid_sentences.append(sentence)
        sentence = []
    first = 0
    sentence.append((row.words, row.pos))
valid_sentences.append(sentence)

#greedy_decoding function computes the state sequence for our HMM Model using the greedy decoding technique
def greedy_decoding(trans_prob, em_prob, prior_prob, valid_sentences, tags):
    sequences = []
    total_score = []
    for sen in valid_sentences:
        prev_tag = None
        seq = []
        score = []
        for i in range(len(sen)):
            best_score = -1
            for j in range(len(tags)):
                state_score = 1
                if i == 0:
                    state_score *= prior_prob[tags[j]]
                else:
                    if str("(" + prev_tag  + "," + tags[j] + ")") in trans_prob:
                        state_score *= trans_prob["(" + prev_tag  + "," + tags[j] + ")"]
                
                if str("(" + tags[j] + ", " + sen[i][0] + ")") in em_prob:
                    state_score *= em_prob["(" + tags[j] + ", " + sen[i][0] + ")"]
                else:
                    state_score *= em_prob["(" + tags[j] + ", " + "<unk>" + ")"]
                
                if(state_score > best_score):
                    best_score = state_score
                    highest_prob_tag = tags[j]
                    
            prev_tag = highest_prob_tag
            seq.append(prev_tag)
            score.append(best_score)
        sequences.append(seq)
        total_score.append(score)

    return sequences, total_score

sequences, total_score = greedy_decoding(trans_prob, em_prob, prior_prob, valid_sentences, tags)

#measure_acc function computes the accuracy of our model by comparing the the predicted tag sequence 
#with groundtruth tag sequence
def measure_acc(sequences, valid_sentences):
    count = 0
    corr_tag_count = 0
    for i in range(len(valid_sentences)):
        for j in range(len(valid_sentences[i])):

            if(sequences[i][j] == valid_sentences[i][j][1]):
                corr_tag_count += 1
            count +=1
    
    acc = corr_tag_count / count
    return acc

print('Accuracy of our Greedy Decoding HMM model on Development data: {}'.format(measure_acc(sequences, valid_sentences)))

#reading the test file
test_data = pd.read_csv("./data/test", sep = '\t', names = ['id', 'words'])
test_data['occ'] = test_data.groupby('words')['words'].transform('size')
test_data['words'] = test_data.apply(lambda row : replace(row), axis = 1)

#storing the test data in a list( called sentences) consisting of lists(called sentence), 
#where each sublist consists of tuples, where each tuple corresponds to one data row in test data
test_sentences = []
sentence = []
first = 1
for row in test_data.itertuples():
    if(row.id == 1 and first == 0):
        test_sentences.append(sentence)
        sentence = []
    first = 0
    sentence.append(row.words)
test_sentences.append(sentence)

test_sequences, test_score = greedy_decoding(trans_prob, em_prob, prior_prob, test_sentences, tags)

#output_file function stores the predicted tag sequence of our HMM model on test dataset in an output file, 
#in the required format
def output_file(test_inputs, test_outputs, filename):
    res = []
    for i in range(len(test_inputs)):
        s = []
        for j in range(len(test_inputs[i])):
            s.append((str(j+1), test_inputs[i][j], test_outputs[i][j]))
        res.append(s)
    
    with open(filename + ".out", 'w') as f:
        for ele in res:
            f.write("\n".join([str(item[0]) + "\t" + item[1] + "\t" + item[2] for item in ele]))
            f.write("\n\n")

#we use the output_file function to store the predictions of greedy decoding by our HMM model 
#on the test data in 'greedy.out'
output_file(test_sentences, test_sequences, "greedy")

#viterbi_decoding function computes the probabilty for each word in a sentence having a tag from the group of all tags
#based on the dynamic programming algorithm called viterbi decoding 
def viterbi_decoding(trans_prob, em_prob, prior_prob, sen, tags):

    n = len(tags)
    viterbi_list = []
    cache = {}
    for si in tags:
        if str("(" + si + ", " + sen[0][0] + ")") in em_prob:
            viterbi_list.append(prior_prob[si] * em_prob["(" + si + ", " + sen[0][0] + ")"])
        else:
            #viterbi_list.append(1)
            viterbi_list.append(prior_prob[si] * em_prob["(" + si + ", " + "<unk>" + ")"])

    for i, l in enumerate(sen):
        word = l[0]
        if i == 0: continue
        temp_list = [None] * n
        for j,tag in enumerate(tags):
            score = -1
            val = 1
            for k, prob in enumerate(viterbi_list):
                if str("(" + tags[k] + "," + tag + ")") in trans_prob and str("(" + tag + ", " + word + ")") in em_prob:
                    val = prob * trans_prob["(" + tags[k] + "," + tag + ")"] * em_prob["(" + tag + ", " + word + ")"]
                else:
                   # val = 1
                   val = prob * trans_prob["(" + tags[k] + "," + tag + ")"] * em_prob["(" + tag + ", " + "<unk>" + ")"]
                if(score < val):
                    score = val
                    cache[str(i) + ", " + tag] = [tags[k], val]
            temp_list[j] = score
        viterbi_list = [x for x in temp_list]
    
    return cache, viterbi_list


c = []
v = []
#we apply viterbi decoding function to all sentences in Validation data
for sen in valid_sentences:
    a, b = viterbi_decoding(trans_prob, em_prob, prior_prob, sen, tags)
    c.append(a)
    v.append(b)

#viterbi_backward function than finds the best possible tag sequence for each sentence based on the 
#probabilites calculated by the viterbi decoding function
def viterbi_backward(tags, cache, viterbi_list):

    num_states = len(tags)
    n = len(cache) // num_states
    best_sequence = []
    best_sequence_breakdown = []
    x = tags[np.argmax(np.asarray(viterbi_list))]
    best_sequence.append(x)

    for i in range(n, 0, -1):
        val = cache[str(i) + ', ' + x][1]
        x = cache[str(i) + ', ' + x][0]
        best_sequence = [x] + best_sequence
        best_sequence_breakdown =  [val] + best_sequence_breakdown
    
    return best_sequence, best_sequence_breakdown

#we apply viterbi backward function to all sentences in Validation data
best_seq = []
best_seq_score = []
for cache, viterbi_list in zip(c, v):

    a, b = viterbi_backward(tags, cache, viterbi_list)
    best_seq.append(a)
    best_seq_score.append(b)

print('Accuracy of our Viterbi Decoding HMM model on Development data: {}'.format(measure_acc(best_seq, valid_sentences)))

#applying the viterbi decoding algorithm on all sentences in the test data using the functions viterbi_decoding
#and viterbi_backward
c = []
v = []

for sen in test_sentences:
    a, b = viterbi_decoding(trans_prob, em_prob, prior_prob, sen, tags)
    c.append(a)
    v.append(b)

best_seq = []
best_seq_score = []
for cache, viterbi_list in zip(c, v):

    a, b = viterbi_backward(tags, cache, viterbi_list)
    best_seq.append(a)
    best_seq_score.append(b)

#we use the output_file function to store the predictions of Viterbi decoding by our HMM model 
#on the test data in 'viterbi.out'
output_file(test_sentences, best_seq, 'viterbi')