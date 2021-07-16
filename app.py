from flask import Flask, request, render_template
import nlpaug
import random
import string
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nltk
from textaugment import EDA, Wordnet
import emoji

app = Flask(__name__)

# home page
@app.route('/index')
def my_form():
    return render_template('index.html',input_text="")

# form view
@app.route('/index', methods=['POST'])
def my_form_post():
    text = request.form['text']
    result = []
    if request.form['group1'] == 'Positive':
        t = EDA()
        t1 = Wordnet()
        result = [
            t.random_swap(text),
            t.random_deletion(text, p=0.3),
            t.random_insertion(text),
            naw.SynonymAug(aug_src='wordnet').augment(text, n=1),
            t1.augment(text),
            nac.OcrAug().augment(text, n=1),
            nac.KeyboardAug().augment(text, n=1),
            nac.RandomCharAug('insert').augment(text, n=1),
            nac.RandomCharAug('swap').augment(text, n=1),
            nac.RandomCharAug('delete').augment(text, n=1),
        ]

    else:
        result = evaluate_negative_augmentation(text)

    return render_template('index.html', result=result, input_text=text)


# helper functions are below

def evaluate_negative_augmentation(text):
    t = EDA()
    words = text.split(" ")
    half_txt = " ".join(words[:int(len(words)/2)])
    rem_txt = " ".join(words[int(len(words)/2):])
    n = int(len(words)/2)
    result = []

    #0. replace with emojis
    result.append(text_to_emoji(text))
    #1. make antonym of whole text
    result.append(naw.AntonymAug().augment(text, n=1))
    #2. insert n words in the half sentence, where n = half of size of sentence
    rand_index = random.randint(0,n)
    result.append(t.random_insertion(sentence=words[rand_index], n=n)+ " " +rem_txt)
    #3. make antonym of whole text and insert a special character at any position
    result.append(get_antonym_with_special_char(t, words, len(words)))
    #4. swap half of the sentence
    result.append(t.random_swap(half_txt)+ " " +rem_txt)
    #5. make half sentence antonym
    result.append(naw.AntonymAug().augment(half_txt, n=1)+ " " +rem_txt)
    #6. insert one random word in half text
    result.append(t.random_insertion(half_txt)+ " " +rem_txt)
    #7. antonym of half and insert random char in another half
    result.append(naw.AntonymAug().augment(half_txt, n=1) + " " + nac.RandomCharAug('insert').augment(rem_txt, n=1))
    #8. antonym of half and swap char in other half
    result.append(naw.AntonymAug().augment(rem_txt, n=1) + " " + nac.RandomCharAug('swap').augment(half_txt, n=1))
    #9. antonym of half and swap word in another half
    result.append(naw.AntonymAug().augment(rem_txt, n=1)+ " " +t.random_swap(half_txt),)
    return result


def get_antonym_with_special_char(t, words, n):
    """
    replace char in antonym augmentation
    """
    s = " ".join(naw.AntonymAug().augment(words, n=1))
    index = random.randint(0,n)
    return s[:index] + random.choice(string.punctuation)  + s[index + 1:]


def text_to_emoji(text):
    """
    Replaces words with possible emojis.
    """
    text = text.replace(",","").replace(".","")
    new_sentence = " ".join([":"+s+":" for s in text.split(" ")])
    emojized =  emoji.emojize(new_sentence, use_aliases=True).split(" ")

    sent = []
    for each in emojized:
        if each in emoji.UNICODE_EMOJI['en']:
            sent.append(each)
        else:
            sent.append(each.replace(":", ""))
    return " ".join(sent)


if __name__ == "__main__":
    app.run(debug=True)