from django.shortcuts import render
from joblib import load
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
categories = ['sci.med', 'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)

# Create your views here.
def index(req):
    model = load('./chatgroup/static/chatgroup.model')
    acc = load('./chatgroup/static/acc.model')
    label = ""
    chat  = ""

    if req.method == 'POST':
        print("POST IN")
        chat = str(req.POST['chat'])
        predict = model.predict([chat])
        label = train.target_names[predict[0]]

    if label == "":
        label = "Unknown."
    return render(req, 'chatgroup/index.html' ,{
            'label':label,
            'acc':acc,
    })