from django.shortcuts import render
import torch
import spacy
from django.conf import settings
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
        context = { "faketext" : prediction.item() }
        return render(request,'basicform.html',context)
    return render(request,'basicform.html')

