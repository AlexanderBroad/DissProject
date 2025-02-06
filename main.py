from flask import Flask, request
import nltk
import newspaper
import textwrap
import tldextract
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import html

app = Flask(__name__)

def initialize_nltk():
    """Download required NLTK data"""
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

def get_publication_details(url):
    """
    Extract publication details from the URL with proper error handling`
    """
    try:
        # Use tldextract to get domain information
        ext = tldextract.extract(url)
        
        # Default values
        details = {
            "name": "Unknown Publication",
        }
        
        # Only update if we have valid domain information
        if ext.domain:
            details["name"] = ext.domain.title()
        return details
    
    except Exception as e:
        print(f"Error in get_publication_details: {str(e)}")  # For debugging
        return {
            "name": "Unknown Publication",
        }

def format_date(date):
    if date is None:
        return "No date available"
    try:
        # Convert to dd/mm/yyyy format
        return date.strftime("%d/%m/%Y")
    except Exception:
        # Fallback if formatting fails
        return str(date)

def analyze_sentiment(text):
    """
    Analyze sentiment of text and return both highlighted and plain versions
    """
    if not text:
        return "", ""
        
    try:
        sia = SentimentIntensityAnalyzer()
        sentences = sent_tokenize(text)
        highlighted_text = []
        plain_text = []

        for sentence in sentences:
            scores = sia.polarity_scores(sentence)
            compound_score = scores['compound']
            
            if compound_score >= 0.05:
                color = "#90EE90"  # Light green for positive
                label = "positive"
            elif compound_score <= -0.05:
                color = "#FFB6C1"  # Light red for negative
                label = "negative"
            else:
                color = "#F0F8FF"  # Light blue for neutral
                label = "neutral"

            highlighted_sentence = f'<span style="background-color: {color};" title="{label} ({compound_score:.2f})">{html.escape(sentence)}</span>'
            highlighted_text.append(highlighted_sentence)
            plain_text.append(html.escape(sentence))

        return " ".join(highlighted_text), " ".join(plain_text)
    except Exception as e:
        print(f"Error in analyze_sentiment: {str(e)}")
        return html.escape(text) if text else "", html.escape(text) if text else ""
    
def get_article_data_from(url):
    try:
        initialize_nltk()
        
        # Get publication details with error checking
        pub_details = get_publication_details(url)
        if not isinstance(pub_details, dict):
            pub_details = {"name": "Unknown Publication"}
        
        article = newspaper.Article(url)
        article.download()
        article.parse()
        article.nlp()
        
        # Safely get article text and summary
        article_text = article.text if article.text else "No article text available"
        article_summary = article.summary if article.summary else "No summary available"

        # Get both highlighted and plain versions of the text
        highlighted_text, plain_text = analyze_sentiment(article_text)
        highlighted_summary, plain_summary = analyze_sentiment(article_summary)
        
        # Create the HTML output with error checking for each component
        output = f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 20px; line-height: 1.6;">
            <div style="border-bottom: 2px solid #333; margin-bottom: 20px;">
                <h2>ARTICLE DETAILS</h2>
            </div>
            
            <div style="margin-bottom: 15px;">
                <strong>🌐 PUBLICATION:</strong><br>
                {html.escape(pub_details.get('name', 'Unknown'))}<br>
            </div>

            <div style="margin-bottom: 15px;">
                <strong>🗞️ TITLE:</strong><br>
                {html.escape(article.title)}<br>
            </div>
            
            <div style="margin-bottom: 15px;">
                <strong>🖋️ AUTHORS:</strong><br>
                {('<br>'.join(f'• {html.escape(author)}' for author in article.authors)) if article.authors else 'No authors found'}
            </div>
            
            <div style="margin-bottom: 15px;">
                <strong>📅 PUBLISH DATE:</strong><br>
                {format_date(article.publish_date)}
            </div>
            
            <div style="margin-bottom: 15px;" id="sentiment-controls">
                <div id="sentiment-legend" style="opacity: 0.3; margin-bottom: 10px; transition: opacity 0.3s ease;">
                    <strong>📊 SENTIMENT LEGEND:</strong><br>
                    <span style="background-color: #90EE90; padding: 0 5px;">Positive</span>
                    <span style="background-color: #F0F8FF; padding: 0 5px; margin: 0 10px;">Neutral</span>
                    <span style="background-color: #FFB6C1; padding: 0 5px;">Negative</span>
                </div>
                
                <label class="switch">
                    <input type="checkbox" id="sentimentToggle">
                    <span class="slider round"></span>
                </label>
            </div>
            
            <div style="margin-bottom: 15px;">
                <strong>📋 SUMMARY:</strong><br>
                <div style="margin-top: 10px;">
                    <div class="text-plain">{plain_summary}</div>
                    <div class="text-highlighted" style="display: none;">{highlighted_summary}</div>
                </div>
            </div>

            <div style="margin-bottom: 15px;">
                <strong>📄 ARTICLE TEXT:</strong><br>
                <div style="margin-top: 10px;">
                    <div class="text-plain">{plain_text}</div>
                    <div class="text-highlighted" style="display: none;">{highlighted_text}</div>
                </div>
            </div>
        </div>
        """
        return output
    
    except Exception as e:
        print(f"Error in get_article_data_from: {str(e)}")  # For debugging
        return f"Error extracting article: {str(e)}"
    
#@app.route("/")
#def index():
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


#def get_article_data_from(url):
    try:
        # Download and parse NLTK resources with punkt pre-trained tokenizer
        nltk.download('punkt', quiet=True)
        
        article = newspaper.article(url)
        article.download()
        article.parse()
        article.nlp()
        
        # Prepare formatted output
        output = []
        
        # Article Details Heading
        output.append("=" * 50)
        output.append(f"ARTICLE DETAILS".center(50))
        output.append("=" * 50)

        # Title
        output.append(f"\n📰 TITLE:")
        output.append(f"   {article.title}")
        
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

#if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)

@app.route("/")
def index():
    url = request.args.get("url", "")
    if url:
        extracted_article_data = get_article_data_from(url)
    else:
        extracted_article_data = ""
    
    return f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            
            /* Toggle Switch Styles */
            .switch {{
                position: relative;
                display: inline-block;
                width: 60px;
                height: 33px;
            }}
            
            .switch input {{
                opacity: 0;
                width: 0;
                height: 0;
            }}
            
            .slider {{
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #ccc;
                transition: .4s;
            }}
            
            .slider:before {{
                position: absolute;
                content: "";
                height: 26px;
                width: 26px;
                left: 4px;
                bottom: 4px;
                background-color: white;
                transition: .4s;
            }}
            
            input:checked + .slider {{
                background-color: #4CAF50;
            }}
            
            input:checked + .slider:before {{
                transform: translateX(26px);
            }}
            
            .slider.round {{
                border-radius: 34px;
            }}
            
            .slider.round:before {{
                border-radius: 50%;
            }}
        </style>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const toggle = document.getElementById('sentimentToggle');
                if (toggle) {{
                    toggle.addEventListener('change', function() {{
                        const legend = document.getElementById('sentiment-legend');
                        const plainTexts = document.getElementsByClassName('text-plain');
                        const highlightedTexts = document.getElementsByClassName('text-highlighted');
                        
                        // Toggle legend opacity
                        legend.style.opacity = this.checked ? '1' : '0.3';
                        
                        // Toggle text versions
                        for (let text of plainTexts) {{
                            text.style.display = this.checked ? 'none' : 'block';
                        }}
                        for (let text of highlightedTexts) {{
                            text.style.display = this.checked ? 'block' : 'none';
                        }}
                    }});
                }}
            }});
        </script>
    </head>
    <body>
        <form action="" method="get" style="max-width: 600px; margin: 20px;">
            <label for="url">Enter/Paste Article URL here:</label><br>
            <input type="url" id="url" name="url" style="width: 100%; padding: 10px; margin: 10px 0;"><br>
            <input type="submit" value="Analyse Article" style="padding: 10px; background-color: #4CAF50; color: white; border: none;">
        </form>
        {extracted_article_data}
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)