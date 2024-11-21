from flask import Flask
from flask import request
import nltk
from newspaper import Article

app = Flask(__name__)

@app.route("/")
def index():
    url = request.args.get("url", "")
    if url:
     processed_article_text = process_article_from(url)
    else:
     processed_article_text = ""
    return (
        """<form action="" method="get">
                Enter/Paste Article URL here: <input type="url" name="url">
                <input type="submit" value="Analyse Article">
            </form>"""
        + "Output: "
        + processed_article_text
    )

def process_article_from(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        nltk.download('punkt_tab')
        article.nlp()
        authors = article.authors
        publish_date = article.publish_date
        article_text = article.text
        summary = article.summary

        article_data = {}
        article_data["authors"] = str(authors)
        article_data["publish_date"] = str(publish_date)
        article_data["article_text"] = article_text
        #article_data["summary"] = summary


        result = " "
        for key in article_data:
            result += article_data[key]

        return result
    except ValueError:
        return "invalid input"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)