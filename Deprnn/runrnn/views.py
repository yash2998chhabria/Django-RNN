from django.shortcuts import render
import torch
import spacy
from django.conf import settings
from nltk.tokenize import sent_tokenize
import nltk 
import torch.nn as nn
import os
# Create your views here.

def displayform(request):
    if request.method == 'POST':
        textcon = request.POST.get('textdata')
        settings.NEW_MODEL.eval()
        tokenized = [tok.text for tok in settings.NLP.tokenizer(textcon)]
        indexed = [settings.NEW_TEXT.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed)
        tensor = tensor.unsqueeze(1)
        prediction = torch.sigmoid(settings.NEW_MODEL(tensor))
        predicted = prediction.item() 
        sent_tokens = sent_tokenize(textcon)
        numeric_symptoms_sent_list=[]
        for sentence in sent_tokens:
            tokenized = [tok.text for tok in settings.NLP.tokenizer(sentence)]
            indexed = [settings.OWN_TEXT.stoi[t] for t in tokenized]
            tensor = torch.LongTensor(indexed)
            tensor = tensor.unsqueeze(1)
            prediction = torch.sigmoid(settings.OWN_DATA_MODEL(tensor))
            numeric_symptoms_sent_list.append(prediction.item())
        print(numeric_symptoms_sent_list)
        context = { "faketext" : predicted,
                    "list":numeric_symptoms_sent_list
                    }
        return render(request,'basicform.html',context)
    return render(request,'basicform.html')

def checkhome(request):
    if request.method == 'POST':
        textcon = request.POST.get('newtextdata')
        settings.NEW_MODEL.eval()
        tokenized = [tok.text for tok in settings.NLP.tokenizer(textcon)]
        indexed = [settings.NEW_TEXT.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed)
        tensor = tensor.unsqueeze(1)
        class RNN(nn.Module):

            def __init__(self, vocab_size, embedding_dim, hidden_dim,
                        output_dim, n_layers, bidirectional, dropout):

                super().__init__()

                self.embedding = nn.Embedding(vocab_size, embedding_dim)

                self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers,
                                bidirectional = bidirectional, dropout=dropout)

                self.fc = nn.Linear(hidden_dim*2, output_dim)

                self.dropout = nn.Dropout(dropout)


            def forward(self, text):

                embedded = self.dropout(self.embedding(text))

                output, hidden = self.rnn(embedded)

                hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))

                return self.fc(hidden.squeeze(0))

        input_dim = len(settings.NEW_TEXT)

        embedding_dim = 100

        hidden_dim = 20
        output_dim = 1

        n_layers = 2
        bidirectional = True

        dropout = 0.5

        NEW_MODEL = RNN(input_dim,
                    embedding_dim,
                    hidden_dim,
                    output_dim,
                    n_layers,
                    bidirectional,
                    dropout)
        NEW_MODEL.load_state_dict(torch.load(os.path.join(settings.BASE_DIR,'models/deprnnscrapped_state_dic')))
        NEW_MODEL.eval()

        prediction = torch.sigmoid(NEW_MODEL(tensor))
        predicted = prediction.item() 
        # sent_tokens = sent_tokenize(textcon)
        # numeric_symptoms_sent_list={}
        # for sentence in sent_tokens:
        #     tokenized = [tok.text for tok in settings.NLP.tokenizer(sentence)]
        #     indexed = [settings.OWN_TEXT.stoi[t] for t in tokenized]
        #     tensor = torch.LongTensor(indexed)
        #     tensor = tensor.unsqueeze(1)
        #     prediction = torch.sigmoid(settings.OWN_DATA_MODEL(tensor))
        #     numeric_symptoms_sent_list[sentence]=prediction.item() * 100
        # print(numeric_symptoms_sent_list)
        # context = { "faketext" : predicted,
        #             "list":numeric_symptoms_sent_list.items()
        #             }
        return render(request,'contact.html',{"faketext":predicted})
    return render(request,'home.html')

def checkresults(request):
    return render(request,'contact.html')    
