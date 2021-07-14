from flask import Flask, request, render_template
import nlpaug
import nlpaug.augmenter.word as naw
import nltk

app = Flask(__name__)

@app.route('/index')
def my_form():
    return render_template('index.html',input_text="")

@app.route('/index', methods=['POST'])
def my_form_post():
    text = request.form['text']
    if request.form['group1'] == 'Positive':
        aug = naw.SynonymAug(aug_src='wordnet')
    else:
        aug = naw.AntonymAug()
    result = aug.augment(text,n=2)
    return render_template('index.html', result=result, input_text=text)


if __name__ == "__main__":
    app.run(threaded=True, debug=True)