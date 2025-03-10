from flask import Flask, request
import nltk
import newspaper
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

def filter_authors(authors, publication_name):
    # Make sure publication_name is a string
    if isinstance(publication_name, dict):
        # If it's a dictionary, try to extract a relevant string field
        publication_name = publication_name.get('name', '') or publication_name.get('title', '') or str(publication_name)
    
    # Convert publication name to lowercase for case-insensitive comparison
    pub_name_lower = str(publication_name).lower()
    
    # Common publication words that might appear in author lists
    common_pub_words = ['www.facebook.com', 'news', 'times', 'post', 'daily', 'guardian', 'mail', 
                        'journal', 'chronicle', 'tribune', 'gazette', 'herald', 'bbc']
    
    filtered_authors = []
    seen_authors = set()  # To track normalized versions of authors already added
    for author in authors:
        # Skip if the author name is empty or just whitespace
        if not author or author.strip() == '':
            continue
            
        # Skip if the author name is the publication name
        if author.lower() == pub_name_lower:
            continue
            
        # Skip if the author name contains the publication name and is less than 35 characters
        # (To avoid filtering out quotes that legitimately contain the publication name)
        if pub_name_lower in author.lower() and len(author) < 35:
            continue
            
        # Skip if the author name is just one of the common publication words
        if author.lower() in common_pub_words:
            continue

        # Create a normalized version of the author name for comparison
        # Remove spaces, hyphens, underscores and convert to lowercase
        normalized_author = author.lower()
        normalized_author = normalized_author.replace(' ', '').replace('-', '').replace('_', '')

        # Skip if we've already seen this author (based on normalized name)
        if normalized_author in seen_authors:
            continue
            
        # Add this normalized version to our seen set
        seen_authors.add(normalized_author)

        filtered_authors.append(author)
    
    return filtered_authors

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
        publication_name = pub_details.get('name', 'Unknown')
        if isinstance(publication_name, dict):
            # Extract the actual name from the dictionary
            publication_string = publication_name.get('name', '')
        else:
            publication_string = publication_name
        
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

        # Filter authors
        filtered_authors = filter_authors(article.authors, publication_string)
        
        # Create the HTML output with error checking for each component
        output = f"""
        <div style="font-family: Arial, sans-serif; font-size:20px; max-width: 800px; margin: 20px; line-height: 1.6;">
            <div style="border-bottom: 2px solid #333; margin-bottom: 15px;">
                <h2>ARTICLE DETAILS</h2>
            </div>
            
            <div style="margin-bottom: 15px;">
                <strong>🌐 PUBLICATION:</strong><br>
                {html.escape(pub_details.get('name', 'Unknown'))}<br>
            </div>

            <div style="margin-bottom: 15px;">
                <strong>🗞️ TITLE:</strong><br>
                <u><a href="{url}" target="_blank">'{html.escape(article.title)}'</a></u><br>
            </div>
            
            <div style="margin-bottom: 15px;">
                <strong>🖋️ AUTHOR(S):</strong><br>
                {('<br>'.join(f'• {html.escape(author)}' for author in filtered_authors)) if filtered_authors else 'No authors found'}
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
            body {{ font-family: Arial, sans-serif; margin: 20px;  }}
            
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

                // Add fade-in effect for results
                const resultsDiv = document.getElementById("results");
                if (resultsDiv && resultsDiv.innerHTML.trim() !== "") {{
                    setTimeout(function() {{
                        resultsDiv.style.opacity = 1;
                    }}, 100);
                }}    
           }});
        </script>
    </head>
    <body style="background-color: moccasin; display: flex; flex-direction: column; align-items: center; text-align: justify; justify-content: center; min-height: 100vh; margin: 0; font-family: Arial, sans-serif;">
            <form action="" method="get" style="width: 100%; max-width: 600px; margin: 20px; text-align: center;">
                <label for="url"><strong>ENTER ARTICLE URL BELOW:</strong></label><br>
                <input type="url" id="url" name="url" style="width: 100%; padding: 10px; margin: 20px 0; strong;"><br>
                <input type="submit" value="ANALYSE ARTICLE" style="padding: 10px; background-color: #4CAF50; color: black; border: 3px black; cursor: pointer; onclick="showResults(event)">
            </form>
    
        <div id="results" style="width: 100%; max-width: 800px; margin: 20px; opacity: 0; transition: opacity 1s ease-in-out;">
        {extracted_article_data}
        </div>

        <button onclick="window.scrollTo({{ top: 0, behavior: 'smooth' }})" id="backToTopBtn" style="position: fixed; bottom: 20px; right: 20px; padding: 10px 15px; background-color: #555; color: white; border: none; border-radius: 4px; cursor: pointer; display: none;">Back to Top</button>

        <script>
            // Back to top button functionality
            window.onscroll = function() {{
                const mybutton = document.getElementById("backToTopBtn");
                if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {{
                    mybutton.style.display = "block";
                }} else {{
                    mybutton.style.display = "none";
                }}
            }};
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)