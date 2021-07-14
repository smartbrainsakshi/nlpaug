from flask import Flask, request, render_template
import nlpaug
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nltk

app = Flask(__name__)

@app.route('/index')
def my_form():
    return render_template('index.html',input_text="")

@app.route('/index', methods=['POST'])
def my_form_post():
    text = request.form['text']
    result = []
    if request.form['group1'] == 'Positive':
        aug = naw.SynonymAug(aug_src='wordnet')
        result = aug.augment(text,n=2)
    else:
        kaug = nac.KeyboardAug()
        result.append(kaug.augment(text,n=1))
        aug = naw.AntonymAug()
        result.append(aug.augment(text,n=1))
        result.append(kaug.augment(result[1],n=1))

    return render_template('index.html', result=result, input_text=text)


if __name__ == "__main__":
    app.run(debug=True)