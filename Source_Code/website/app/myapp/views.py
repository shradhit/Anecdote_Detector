import cPickle
import os
import re

import nltk
import pandas as pd
from django.contrib.auth.hashers import check_password, make_password
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from nltk.tag import StanfordNERTagger

from myapp.models import User

# Create your views here.


def init():
    os.environ['STANFORD_PARSER'] = '/home/shirish/stanford-corenlp/stanford-ner-2018-02-27/'
    os.environ['STANFORD_MODELS'] = '/home/shirish/stanford-corenlp/stanford-ner-2018-02-27/'
    os.environ['CLASSPATH'] = "/home/shirish/stanford-corenlp/stanford-ner-2018-02-27/:/home/shirish/stanford-corenlp/stanford-ner-2018-02-27/classifiers:/home/shirish/stanford-corenlp/stanford-ner-2018-02-27/lib"
    print os.curdir
    global anecdote_model
    print "Reading the model"
    anecdote_model = cPickle.load(open('model.pkl', 'rb'))
    print "Read the model"
    global starting_phrases, demonstrative_pronouns, imp_cols
    imp_cols = ["POS", "NNP", "VBD", "VBG", "PRP", "VBZ"]
    starting_phrases = re.compile(
        "^For example|^Once upon a time|^Long time ago|^For instance|^Let's take an example of|^However|^In|^For|^Another Example|^Consider|^Whenever|^Take", re.IGNORECASE)
    demonstrative_pronouns = re.compile(
        "\bSuch\b|\bThis\b|\bThat\b|\bThese\b|\bThose\b|\bEither\b|\bNeither\b|\bHis\b|\bHer\b|\bThey\b|\bIt\b|\bShe\b|\bHe\b|\bThem\b|\bthemselves\b|\bherself\b|\bhimself\b|\bWe\b|\bso\b", re.IGNORECASE)
    print "ST"
    global st
    st = StanfordNERTagger("/home/shirish/stanford-corenlp/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz",
                           "/home/shirish/stanford-corenlp/stanford-ner-2018-02-27/stanford-ner.jar", encoding='utf-8')
    print "Exited from init"


def loginPage(request):
    if request.method == "GET":
        if request.session.get("user", False):
            return redirect("/home")
        else:
            return render(request, "login.html", {"TAG_LOGIN": "None"})
    elif request.method == "POST":

        email1 = request.POST.get("email")
        password1 = request.POST.get("password")
        print "Received", email1, password1
        try:
            b = User.objects.get(email=email1)
            print b
        except:
            b = None
            return render(request, "login.html", {"TAG_LOGIN": "block"})
        print check_password(password1, b.password)
        if not check_password(password1, b.password):
            return render(request, "login.html", {"TAG_LOGIN": "block"})
        else:
            print "Successfully logged in!"
            request.session["user"] = b.id
            return redirect('/home')


def home(request):
    print make_password("plain_text")
    if request.session.get("user", False):
        return render(request, "home.html")
    else:
        return redirect("/login")


def registerPage(request):
    if request.method == "GET":
        return render(request, "register.html")
    else:
        email1 = request.POST["email"]
        password1 = make_password(request.POST["password"])
        b = User(email=email1, password=password1)
        b.save()
        return redirect("/login")


def login(request):
    pass


def logout(request):
    print "Logout"
    del request.session["user"]
    return redirect("/login")


def featureset_df(taggedSents, value, imp_cols):
    """
    This function returns a dataframe consisting of the imp_cols and target column after
    removing the NaN values
    @param taggedSents = list of sentences, where each sentence is a list of POS tagged words
    @param value = Value given to the target column in the dataframe
    Returns = dataframe with columns as imp_cols and target with value @param value
    """
    print "Entered FeatureSet"
    sents1 = []
    print taggedSents[2]
    d = {}
    for i in taggedSents:
        l = {}
        for j in i:
            if j[1].isalpha():
                # If Noun
                l[j[1]] = 1
                d[j[1]] = 1
        sents1.append(l)
    print "After for"
    df = pd.DataFrame(sents1)
    for i in imp_cols:
        if d.get(i, False) == False:
            df[i] = 0
    print df.columns
    df.fillna(0, inplace=True)
    print imp_cols
    # print df.head(2)
    print df.columns
    print df
    features = df[imp_cols]
    print df.head(2)
    # features = df
    features["target"] = value
    print "Exited from featureset_df"
    return features


def process(request):
    global starting_phrases, demonstrative_pronouns, imp_cols
    # print os.getcwd()
    print "Executed"
    s = request.POST.get("data", None)
    if s is None:
        print "Executed in if"
        f = request.FILES['file']
        s = f.read()
    print s
    print "after Executed"
    # Read string from file
    try:
        s = unicode(s, errors="ignore")
    except:
        pass
    print s
    # list of sentences
    sentences = nltk.sent_tokenize(s)
    print sentences
    # list of POS Tagged Sentences
    tagged_sentences = []
    for i in sentences:
        tagged_sentences.append(nltk.pos_tag(nltk.word_tokenize(i)))
    print tagged_sentences[0]
    df = featureset_df(tagged_sentences, 0, imp_cols)
    print "Dataframe"
    print df
    # Array of 0, 1 values
    global anecdote_model
    predicted = anecdote_model.predict(df[imp_cols])
    print "Predictions"
    print list(predicted)
    rule_based = []
    for k, v in enumerate(sentences):
        v = v.strip()
        rule_based.append(0)
        if len(demonstrative_pronouns.findall(v)) == 0:
            if starting_phrases.match(v) is not None:
                rule_based[k] = 2
            else:
                a = st.tag(nltk.word_tokenize(v))
                for i in a:
                    if i[1] == u'PERSON' or i[1] == u"ORGANIZATION":
                        rule_based[k] = 1
                        break
    print "Rule Based"
    print rule_based
    # Heuristics
    # Two arrays are predicted, rule_based
    # Now where both are 1 or the addition of both >= 2 then that is a start of the anecdote
    # If the number of sentences in the anecdotes is less than 3 then it's not an anecdote
    # anecdotes array consists of start and end denoted by 1, -1
    anecdotes = []
    flag = 0
    num_z = 0
    anec_count = 0
    for k, v in enumerate(zip(predicted, rule_based)):
        anecdotes.append(0)
        if v[0] + v[1] >= 2 and flag == 0:
            anec_count += 1
            flag = 1
            anecdotes[k] = 1
        elif flag == 1 and v[0] == 1:
            anec_count += 1
        elif flag == 1 and v[0] == 0:
            num_z += 1
            if num_z >= 2:
                anecdotes[k - 2] = -1
                flag = 0
                num_z = 0
                if anec_count < 2:
                    anecdotes[k - 2] = 0
                anec_count = 0
        print k, v, anec_count, flag
    if flag == 1:
        if anec_count < 2:
            anecdotes[k - 1] = 0
        anecdotes[len(predicted) - 1] = -1
    print "Anecdotes"
    print anecdotes
    global s1
    s1 = ""
    h1 = ""
    for k, i in enumerate(anecdotes):
        if i == 1:
            s1 += "<story> "
            h1 += '<span style="color:red"> '
            h1 += sentences[k]
            s1 += sentences[k]
        elif i == -1:
            s1 += sentences[k]
            h1 += sentences[k]
            h1 += ' </span>'
            s1 += " </story>"
        else:
            h1 += sentences[k]
            s1 += sentences[k]
    return JsonResponse({"output": h1, "input": s})


def download(request):
    response = HttpResponse(s1)
    response['content-type'] = 'application/txt'
    response['Content-Disposition'] = 'attachment;filename=file.txt'
    return response
