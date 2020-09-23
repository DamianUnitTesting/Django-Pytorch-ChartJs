from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.csrf import csrf_exempt

import random
import json
import logging

import torch
from .model import NeuralNet
from .nltk_utils import bag_of_words, tokenize


def index(request):
    return render(request, "index.html")


#@ensure_csrf_cookie
@require_http_methods(["GET", "POST"])
@csrf_exempt
def charbotrequest(request):
    if request.method == 'POST':
        data1 = json.loads(request.body)
        message = data1['responses']
                
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open('intents.json', 'r') as json_data:
            intents = json.load(json_data)

        FILE = "data.pth"
        datatorch = torch.load(FILE)

        input_size = datatorch["input_size"]
        hidden_size = datatorch["hidden_size"]
        output_size = datatorch["output_size"]
        all_words = datatorch['all_words']
        tags = datatorch['tags']
        model_state = datatorch["model_state"]

        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(model_state)
        model.eval()

        bot_name = "ChartBot"
        
        while True:              
            sentence = message
            sentence = tokenize(sentence)
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)

            output = model(X)
            _, predicted = torch.max(output, dim=1)

            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        chat_response = (f"{random.choice(intent['responses'])}")

                        data = {   
                        "text":chat_response,
                        "sender":bot_name,          
                        }
                                                        
                        return HttpResponse(json.dumps(data),content_type='application/json')
 
            else:
                print(f"{bot_name}: I do not understand...")
                return HttpResponse(data)
        

@require_http_methods(["GET", "POST"])
@csrf_exempt
def chartbot(request):
    return render(request, "chartbot.html")

      
