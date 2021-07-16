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
            ["Random word swap: ", t.random_swap(text)],
            ["Random word delete: ", t.random_deletion(text, p=0.3)],
            ["Random word insert: ", t.random_insertion(text)],
            ["Synonym Augmentation: ", naw.SynonymAug(aug_src='wordnet').augment(text, n=1)],
            ["OCR Augmentation: ", nac.OcrAug().augment(text, n=1)],
            ["KeyBoard Augmentation: ", nac.KeyboardAug().augment(text, n=1)],
            ["Random Char insert", nac.RandomCharAug('insert').augment(text, n=1)],
            ["Random Char swap", nac.RandomCharAug('swap').augment(text, n=1)],
            ["Random Char delete", nac.RandomCharAug('delete').augment(text, n=1)],
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
    result.append(["Text to emoji: ", text_to_emoji(text)])
    #1. make antonym of whole text
    result.append(["Antonym of text: ", naw.AntonymAug().augment(text, n=1)])
    #2. insert n words in the half sentence, where n = half of size of sentence
    try:
        rand_index = random.randint(0,n)
        result.append(["Insert words in first half of sentence: ", t.random_insertion(sentence=words[rand_index], n=n)+ " " +rem_txt])
    except:
        pass
    #3. make antonym of whole text and insert a special character at any position
    result.append(["Special character insertion: ", get_with_special_char(text)])
    #4. swap half of the sentence
    result.append(["Swap in the first half of sentence: ", t.random_swap(half_txt)+ " " +rem_txt])
    #5. make half sentence antonym
    result.append(["Antonym of half sentence: ", naw.AntonymAug().augment(half_txt, n=1)+ " " +rem_txt])
    #6. insert one random word in half text
    result.append(["Random one word insertion in first half: ", t.random_insertion(half_txt)+ " " +rem_txt])
    #7. antonym of half and insert random char in another half
    result.append(["Antonym of first half and random character in second half: ", naw.AntonymAug().augment(half_txt, n=1) + " " + nac.RandomCharAug('insert').augment(rem_txt, n=1)])
    return result


def get_with_special_char(text):
    """
    replace char in text
    """
    # get random indexes to be replaced with special characters which will be minimum()
    indexes = random.sample(range(0, len(text)), min(round(len(text)/2), 15))
    for index in indexes:
        text = text[:index] + random.choice(string.punctuation) + text[index + 1:]

    return text


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