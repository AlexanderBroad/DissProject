from flask import Flask, request
import nltk
import newspaper
import textwrap

app = Flask(__name__)

@app.route("/")
def index():
    url = request.args.get("url", "")
    if url:
     extracted_article_data = get_article_data_from(url)
    else:
     extracted_article_data = ""
    return (
        """<form action="" method="get" style="font-family: Arial, sans-serif; max-width: 800px; margin: 20px;">
                Enter/Paste Article URL here: <input type="url" name="url" style="width: 100%; padding: 10px; margin: 10px 0;">
                <input type="submit" value="Analyse Article" style="padding: 10px; background-color: #4CAF50; color: white; border: none;">
            </form>"""
        + "<pre style='white-space: pre-wrap; word-wrap: break-word; font-family: Verdana, sans-serif; max-width: 800px; margin: 20px; line-height: 2;'>" 
        + extracted_article_data 
        + "</pre>"
    )

def format_date(date):
    if date is None:
        return "No date available"
    try:
        # Convert to dd/mm/yyyy format
        return date.strftime("%d/%m/%Y")
    except Exception:
        # Fallback if formatting fails
        return str(date)

def get_article_data_from(url):
    try:
        # Download and parse NLTK resources
        nltk.download('punkt', quiet=True)
        
        article = newspaper.article(url)
        article.download()
        article.parse()
        article.nlp()
        
        # Prepare formatted output
        output = []
        
        # Title
        output.append("=" * 50)
        output.append(f"ARTICLE DETAILS".center(50))
        output.append("=" * 50)
        
        # Authors
        if article.authors:
            output.append(f"\n🖋️ AUTHORS:")
            for author in article.authors:
                output.append(f"   • {author}")
        else:
            output.append("\n🖋️ AUTHORS: No authors found")

        # Publish Date (formatted)
        output.append(f"\n📅 PUBLISH DATE:")
        output.append(f"   {format_date(article.publish_date)}")

        # Summary
        output.append("\n📋 SUMMARY:")
        wrapped_summary = textwrap.fill(article.summary, width=80, initial_indent='   ', subsequent_indent='   ')
        output.append(wrapped_summary)
        
        # Article Text
        output.append("\n📄 ARTICLE TEXT:")
        # Wrap text to 80 characters for better readability
        wrapped_text = textwrap.fill(article.text, width=80, initial_indent='   ', subsequent_indent='   ')
        output.append(wrapped_text)
        
        # Final formatting
        output.append("\n" + "=" * 50)
        
        return "\n".join(output)
    
    except Exception as e:
        return f"Error extracting article: {str(e)}"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)