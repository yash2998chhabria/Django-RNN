from django.shortcuts import render
import torch
import spacy
from django.conf import settings
from nltk.tokenize import sent_tokenize
import nltk 
from .models import User_Data
# Create your views here.

def displayform(request):
    if request.method == 'POST':
        textcon = request.POST.get('textdata')
        settings.NEW_MODEL.eval()
        tokenized = [tok.text for tok in settings.NLP.tokenizer(textcon)]
        indexed = [settings.NEW_TEXT.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed)
        tensor = tensor.unsqueeze(1)
        tobesig = settings.NEW_MODEL(tensor)
        prediction = torch.sigmoid(tobesig)
        predicted = prediction.item() 
        sent_tokens = sent_tokenize(textcon)
        numeric_symptoms_sent_list=[]
        for sentence in sent_tokens:
            tokenized = [tok.text for tok in settings.NLP.tokenizer(sentence)]
            indexed = [settings.OWN_TEXT.stoi[t] for t in tokenized]
            tensor = torch.LongTensor(indexed)
            tensor = tensor.unsqueeze(1)
            tobesig = settings.OWN_DATA_MODEL(tensor)
            prediction = torch.sigmoid(tobesig)
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
        checkbox_val = request.POST.get('checkbox')
        tokenized = [tok.text for tok in settings.NLP.tokenizer(textcon)]
        indexed = [settings.NEW_TEXT.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed)
        tensor = tensor.unsqueeze(1)
        prediction = torch.sigmoid(settings.NEW_MODEL(tensor))
        predicted = float(str(prediction.item() * 100)[:4])
        sent_tokens = sent_tokenize(textcon)
        numeric_symptoms_sent_list={}
        data_base_arr = []
        for sentence in sent_tokens:
            tokenized = [tok.text for tok in settings.NLP.tokenizer(sentence)]
            indexed = [settings.OWN_TEXT.stoi[t] for t in tokenized]
            tensor = torch.LongTensor(indexed)
            tensor = tensor.unsqueeze(1)
            prediction = torch.sigmoid(settings.OWN_DATA_MODEL(tensor))
            numeric_symptoms_sent_list[sentence]= float(str(prediction.item() * 100)[:4])
            data_base_arr.append(float(str(prediction.item() * 100)[:4]))
        context = { "faketext" : predicted,
                    "list":numeric_symptoms_sent_list.items()
                    }
        if checkbox_val:
            new_data = User_Data.objects.create(text=textcon,total_result=predicted,sent_result=data_base_arr)
            new_data.save()

        return render(request,'contact.html',context)
    return render(request,'home.html')

def checkresults(request):
    return render(request,'contact.html')

def aboutpage(request):
    return render(request,'about.html')
