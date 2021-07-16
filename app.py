from flask import Flask, request, render_template
import nlpaug
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nltk
from textaugment import EDA

app = Flask(__name__)

@app.route('/index')
def my_form():
    return render_template('index.html',input_text="")

@app.route('/index', methods=['POST'])
def my_form_post():
    text = request.form['text']
    result = []
    if request.form['group1'] == 'Positive':
        t = EDA()
        result = [
            t.random_swap(text),
            t.random_deletion(text, p=0.2),
            t.random_insertion(text),
        ]
    else:
        aug = nac.OcrAug()
        result = []
        result.append(aug.augment(text, n=1))
        aug = nac.KeyboardAug()
        result.append(aug.augment(text, n=1))

    return render_template('index.html', result=result, input_text=text)


if __name__ == "__main__":
    app.run(debug=True)