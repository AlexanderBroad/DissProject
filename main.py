from flask import Flask, request, render_template_string, url_for
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from urllib.parse import urlparse
import nltk
import newspaper
import tldextract
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from NewsSentiment import TargetSentimentClassifier
from nltk.tokenize import sent_tokenize
import re
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


    """
    Analyze sentiment of text using NewsSentiment and return both highlighted and plain versions
    """
    if not text:
        return "", ""
        
    try:
        # Load NER model
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

        # Use aggregation_strategy to get world_level entities
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        
        # Get named entities
        ner_results = nlp(text)

        # Initialise target sentiment classifier
        tsc = TargetSentimentClassifier()

        # Store entity sentiment data
        entity_sentiments = {}

        # Create a list of (entity, start, end, sentiment) tuples to process later
        entity_data = []

        # Process each entity
        for entity in ner_results:
            entity_text = entity['word']
            start = entity['start']
            end = entity['end']

            # Skip invalid entities
            if start is None or end is None or start >= end:
                continue

            # Get left context, target, and right context for NewsSentiment
            left_context = text[:start]
            target = text[start:end]
            right_context = text[end:]

            # Get sentiment
            try:
                # The output is a tuple of dictionaries with sentiment scores
                sentiment_result = tsc.infer_from_text(left_context, target, right_context)

                # Extract sentiment data
                sentiment_label = sentiment_result[0]['class_label']
                confidence = sentiment_result[0]['class_prob']

            except Exception as inner_e:
                print(f"Error in sentiment analysis for entity '{target}': {str(inner_e)}")
                # Default to neutral if sentiment analysis fails
                sentiment_label = "neutral"
                confidence = 0.5

            # Store entity info for later processing
            entity_data.append((target, start, end, sentiment_label, confidence))

            # Store in our results dictionary
            if target not in entity_sentiments:
                entity_sentiments[target] = {
                    'sentiment': sentiment_label,
                    'confidence': confidence,
                    'entity_type': entity['entity_group']
                }
        
        # Sort entities by their position in text to handle overlapping entities
        entity_data.sort(key=lambda x: x[1])
        
        # Create highlighted HTML version
        highlighted_text = ""
        plain_text = html.escape(text)  # This is for the plain version
        last_pos = 0
        
        for entity, start, end, sentiment, confidence in entity_data:
            # Add text before entity
            highlighted_text += html.escape(text[last_pos:start])
            
            # Determine colour based on sentiment
            if sentiment == "positive":
                color = "#90EE90"  # Light green for positive
            elif sentiment == "negative":
                color = "#FFB6C1"  # Light red for negative
            else:
                color = "#F0F8FF"  # Light blue for neutral
            
            # Add highlighted entity with tooltip
            entity_span = f'<span style="background-color: {color}; font-weight: bold;" title="{sentiment} (confidence: {confidence:.2f})">{html.escape(entity)}</span>'
            highlighted_text += entity_span
            
            last_pos = end
        
        # Add remaining text
        highlighted_text += html.escape(text[last_pos:])
        
        return plain_text, highlighted_text, entity_sentiments
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in analyze_entity_sentiments: {str(e)}")
        escaped_text = html.escape(text) if text else ""
        return escaped_text, escaped_text, {}


    """
    Analyze sentiment of text using NewsSentiment and return both highlighted and plain versions
    """
    if not text:
        return "", ""
        
    try:
        # Initialize the NewsSentiment classifier
        classifier = TargetSentimentClassifier()
        
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)
        highlighted_text = []
        plain_text = []
        
        for sentence in sentences:
            # Based on the error, infer() is expecting a specific format with target and context
            # For general sentiment analysis, we can use the sentence as both the text and target
            predictions = classifier.infer([(sentence, sentence)])
            
            # Check if we got valid predictions
            if predictions and len(predictions) > 0:
                sentiment_label = predictions[0].label
                confidence = predictions[0].confidence
            else:
                sentiment_label = "neutral"
                confidence = 0.0
            
            # Map NewsSentiment labels to your color scheme
            if sentiment_label == "positive":
                color = "#90EE90"  # Light green for positive
            elif sentiment_label == "negative":
                color = "#FFB6C1"  # Light red for negative
            else:
                color = "#F0F8FF"  # Light blue for neutral
            
            # Create highlighted version with sentiment info in tooltip
            highlighted_sentence = f'<span style="background-color: {color};" title="{sentiment_label} (confidence: {confidence:.2f})">{html.escape(sentence)}</span>'
            highlighted_text.append(highlighted_sentence)
            plain_text.append(html.escape(sentence))
        
        return " ".join(highlighted_text), " ".join(plain_text)
    
    except Exception as e:
        print(f"Error in analyze_sentiment: {str(e)}")
        # Fallback to original text if there's an error
        return html.escape(text) if text else "", html.escape(text) if text else ""

def analyze_sentiment_newssentiment(text):
    """
    Analyze sentiment of text using NewsSentiment with sentence-level chunking
    for handling long texts and proper entity sentiment analysis
    """
    if not text:
        return "", "", {}
    
    try:
        # Pre-process: remove all-caps sentences that are likely hyperlinks
        sentences = sent_tokenize(text)
        filtered_sentences = []
        
        for sentence in sentences:
            # Check if sentence is all uppercase (allowing for punctuation and spaces)
            words = [w for w in re.findall(r'\w+', sentence) if len(w) > 1]  # Only consider words with 2+ chars
            
            # Skip if sentence is empty after filtering
            if not words:
                continue
                
            # Calculate percentage of all-caps words
            caps_words = [w for w in words if w.isupper()]
            caps_percentage = len(caps_words) / len(words) if words else 0
            
            # Keep sentence if less than 80% of words are all caps
            if caps_percentage < 0.8:
                filtered_sentences.append(sentence)
        
        # Rebuild text without all-caps sentences
        filtered_text = " ".join(filtered_sentences)
        
        # If all text was removed, return empty results
        if not filtered_text.strip():
            return "", "", {}
        
        # Load NER model
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

        # Use aggregation_strategy to get word-level entities
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        
        # Initialize target sentiment classifier
        tsc = TargetSentimentClassifier()

        # Store entity sentiment data across all sentences
        entity_sentiments = {}
        
        # Track overall position in the original text
        current_position = 0
        entity_data_all = []
        
        # Process each sentence separately
        for sentence in filtered_sentences:
            # Skip empty sentences
            if not sentence.strip():
                current_position += len(sentence) + 1  # +1 for the space added between sentences
                continue
            
            # Get the exact position of this sentence in the original text
            sentence_position = filtered_text.find(sentence, current_position)
            if sentence_position == -1:  # Should never happen with proper sentence tokenization
                sentence_position = current_position
            
            # Update current position for next iteration
            current_position = sentence_position + len(sentence)
                
            # Check if sentence length is within model's token limit
            tokens = tokenizer.encode(sentence)
            if len(tokens) > 510:  # Leave room for special tokens
                # For extra long sentences, split into phrases by punctuation
                phrases = []
                current_phrase = ""
                
                for word in sentence.split():
                    if len(tokenizer.encode(current_phrase + " " + word)) <= 510:
                        current_phrase += " " + word if current_phrase else word
                    else:
                        if current_phrase:
                            phrases.append(current_phrase)
                        current_phrase = word
                
                if current_phrase:
                    phrases.append(current_phrase)
                
                # Process each phrase
                for phrase in phrases:
                    if not phrase.strip():
                        continue
                    
                    # Find exact position of this phrase in the original text
                    phrase_position = filtered_text.find(phrase, sentence_position)
                    if phrase_position == -1:  # If not found exactly, use relative positioning
                        phrase_position = sentence_position
                        sentence_position += len(phrase)
                    else:
                        sentence_position = phrase_position + len(phrase)
                    
                    # Process this phrase
                    process_chunk(phrase, phrase_position, nlp, tsc, entity_sentiments, entity_data_all)
            else:
                # Process the sentence normally
                process_chunk(sentence, sentence_position, nlp, tsc, entity_sentiments, entity_data_all)
        
        # Sort entities by their position in text
        entity_data_all.sort(key=lambda x: x[1])
        
        # Create highlighted HTML version with correct positioning
        highlighted_text = ""
        plain_text = html.escape(filtered_text)
        last_pos = 0
        
        # Handle potential overlapping entities
        filtered_entities = []
        for i, (entity, start, end, sentiment, confidence) in enumerate(entity_data_all):
            # Skip this entity if it overlaps with a higher confidence entity we've already included
            should_skip = False
            for prev_entity, prev_start, prev_end, prev_sentiment, prev_confidence in filtered_entities:
                # Check for overlap
                if (start < prev_end and end > prev_start):
                    # Keep only the higher confidence entity
                    should_skip = confidence <= prev_confidence
                    break
            
            if not should_skip:
                filtered_entities.append((entity, start, end, sentiment, confidence))
        
        # Generate highlighted text with non-overlapping entities
        for entity, start, end, sentiment, confidence in filtered_entities:
            # Ensure start and end positions are valid
            if start < 0 or end > len(filtered_text) or start >= end:
                continue
                
            # Add text before entity
            highlighted_text += html.escape(filtered_text[last_pos:start])
            
            # Determine color based on sentiment
            if sentiment == "positive":
                color = "#90EE90"  # Light green for positive
            elif sentiment == "negative":
                color = "#FFB6C1"  # Light red for negative
            else:
                color = "#F0F8FF"  # Light blue for neutral
            
            # Add highlighted entity with tooltip
            entity_span = f'<span style="background-color: {color};" title="{sentiment} (confidence: {confidence:.2f})">{html.escape(filtered_text[start:end])}</span>'
            highlighted_text += entity_span
            
            last_pos = end
        
        # Add remaining text
        highlighted_text += html.escape(filtered_text[last_pos:])
        
        return plain_text, highlighted_text, entity_sentiments
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in analyze_entity_sentiments: {str(e)}")
        escaped_text = html.escape(text) if text else ""
        return escaped_text, escaped_text, {}

def process_chunk(chunk, chunk_position, nlp, tsc, entity_sentiments, entity_data_all):
    """
    Process a single chunk (sentence or phrase) for entity sentiment analysis
    and add results to the overall data structures
    
    Args:
        chunk: Text chunk to process
        chunk_position: Exact position of this chunk in the original text
        nlp: NER pipeline
        tsc: Target sentiment classifier
        entity_sentiments: Dictionary to store entity sentiment information
        entity_data_all: List to store entity position and sentiment data
    """
    # Get named entities for this chunk
    try:
        ner_results = nlp(chunk)
    except Exception as e:
        print(f"Error in NER for chunk: {str(e)}")
        return
    
    # Process each entity in this chunk
    for entity in ner_results:
        entity_text = entity['word']
        
        # Only process if it's a person (PER) or organization (ORG)
        if entity['entity_group'] not in ['PER', 'ORG', 'LOC', 'MISC']:
            continue
            
        # Get entity positions within the chunk
        chunk_start = entity['start']
        chunk_end = entity['end']
        
        # Skip invalid entities
        if chunk_start is None or chunk_end is None or chunk_start >= chunk_end:
            continue
            
        # Calculate positions in the original text
        global_start = chunk_position + chunk_start
        global_end = chunk_position + chunk_end

        # Verify that the extracted text matches between chunk and original text
        chunk_entity = chunk[chunk_start:chunk_end]
        
        # Get context for sentiment analysis
        left_context = chunk[:chunk_start]
        target = chunk_entity
        right_context = chunk[chunk_end:]

        try:
            # Get sentiment for this entity
            sentiment_result = tsc.infer_from_text(left_context, target, right_context)
            sentiment_label = sentiment_result[0]['class_label']
            confidence = sentiment_result[0]['class_prob']
        except Exception as inner_e:
            print(f"Error in sentiment analysis for entity '{target}': {str(inner_e)}")
            sentiment_label = "neutral"
            confidence = 0.5

        # Store entity info with global position
        entity_data_all.append((chunk_entity, global_start, global_end, sentiment_label, confidence))

        # Store in results dictionary
        entity_key = chunk_entity
        if entity_key not in entity_sentiments:
            entity_sentiments[entity_key] = {
                'sentiment': sentiment_label,
                'confidence': confidence,
                'entity_type': entity['entity_group'],
                'occurrences': 1
            }
        else:
            # Update existing entity information
            current = entity_sentiments[entity_key]
            # If this occurrence has higher confidence, update the sentiment
            if confidence > current['confidence']:
                current['sentiment'] = sentiment_label
                current['confidence'] = confidence
            current['occurrences'] = current.get('occurrences', 0) + 1

def generate_top_entities_report(entity_sentiments):
    """
    Generate a report of the top 5 entities by occurrence count with their sentiment breakdowns
    
    Args:
        entity_sentiments: Dictionary of entity sentiment data
        
    Returns:
        List of dictionaries with top entity information
    """
    if not entity_sentiments:
        return []
        
    # Sort entities by occurrence count
    sorted_entities = sorted(
        entity_sentiments.items(), 
        key=lambda x: (x[1]['occurrences'], x[1]['confidence']), 
        reverse=True
    )
    
    # Take top 5 entities
    top_entities = []
    for entity_name, entity_data in sorted_entities[:5]:
        top_entities.append({
            'name': entity_name,
            'type': entity_data['entity_type'],
            'occurrences': entity_data['occurrences'],
            'sentiment': entity_data['sentiment'],
            'confidence': entity_data['confidence']
        })
    
    return top_entities

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
        highlighted_text, plain_text, entity_sentiments = analyze_sentiment_newssentiment(article_text)
        highlighted_summary, plain_summary, _ = analyze_sentiment_newssentiment(article_summary)

        # Generate top 5 entities report
        top_entities = generate_top_entities_report(entity_sentiments)

        # Generate the entities HTML section
        entities_html = generate_entities_html(top_entities)

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

            {entities_html}

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

def generate_entities_html(top_entities):
    """
    Generate HTML for the top entities display
    
    Args:
        top_entities: List of dictionaries with entity information
        
    Returns:
        HTML string for the entities section
    """
    if not top_entities:
        return ""
    
    entities_html = """
    <div style="margin-bottom: 25px;">
        <strong>🔍 TOP ENTITIES:</strong>
        <div style="margin-top: 15px;">
    """
    
    for entity in top_entities:
        # Determine the color based on sentiment
        if entity["sentiment"] == "positive":
            border_color = "#90EE90"  # Light green
            bg_color = "#90EE90"
        elif entity["sentiment"] == "negative":
            border_color = "#FFB6C1"  # Light pink
            bg_color = "#FFB6C1" 
        else:
            border_color = "#F0F8FF"  # Light blue
            bg_color = "#A9D0F5"
        
        # Calculate confidence percentage for display
        confidence_pct = f"{entity['confidence'] * 100:.1f}%"
        confidence_width = f"{entity['confidence'] * 100}%"
        
        # Create the entity cards
        entities_html += f"""
        <div style="border: 1px solid #ddd; border-left: 5px solid {border_color}; border-radius: 8px; 
                  padding: 15px; margin-bottom: 15px; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0;">{html.escape(entity['name'])}</h3>
                <span style="display: inline-block; padding: 3px 8px; border-radius: 12px; 
                           background-color: #f0f0f0; font-size: 0.8em;">{entity['type']}</span>
            </div>
            <p style="margin: 8px 0;">Occurrences: {entity['occurrences']}</p>
            <p style="margin: 8px 0;">Sentiment: {entity['sentiment'].capitalize()}</p>
            <div style="height: 8px; background-color: #e0e0e0; border-radius: 4px; margin-top: 8px;">
                <div style="height: 100%; border-radius: 4px; width: {confidence_width}; 
                          background-color: {bg_color};"></div>
            </div>
            <p style="text-align: right; font-size: 0.8em; margin: 5px 0 0 0;">Confidence: {confidence_pct}</p>
        </div>
        """
    
    entities_html += """
        </div>
    </div>
    """
    
    return entities_html

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
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Article Bias Indicator</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px;  }}

            /* Loading bar styles */
            .loader {{
                border: 16px solid #f3f3f3;
                border-radius: 50%;
                border-top: 16px solid #3498db;
                width: 80px;
                height: 80px;
                animation: spin 2s linear infinite;
                margin: 20px auto;
                display: none;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
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
            
            /* Fade-in transition for results */
            .results-container {{
                opacity: 0;
                transition: opacity 1s ease-in-out;
            }}
        </style>

        <script>
                // Sentiment Toggle switch
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
                                text.style.display = this.checked ? 'block' : 'none';
                            }}
                            for (let text of highlightedTexts) {{
                                text.style.display = this.checked ? 'none' : 'block';
                            }}
                        }});
                    }}
                    
                    // Handle form submission with loading animation
                    const form = document.getElementById('analysisForm');
                    if (form) {{
                        form.addEventListener('submit', function(e) {{
                            showLoader();
                        }});
                    }}
                    
                    // Auto show results if URL is already in the form
                    const url = document.getElementById('url').value;
                    if (url) {{
                        const resultsContainer = document.getElementById('resultsContainer');
                        if (resultsContainer) {{
                            setTimeout(function() {{
                                resultsContainer.style.opacity = '1';
                            }}, 300);
                        }}
                    }}
                }});
                
                // Back to top button functionality
                window.onscroll = function() {{
                    const mybutton = document.getElementById("backToTopBtn");
                    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {{
                        mybutton.style.display = "block";
                    }} else {{
                        mybutton.style.display = "none";
                    }}
                }};

                // Show loading indicator and hide results
                function showLoader() {{
                    document.getElementById("loader").style.display = "block";
                    const resultsContainer = document.getElementById("resultsContainer");
                    if (resultsContainer) {{
                        resultsContainer.style.opacity = "0";
                    }}
                    return true;
                }}
        </script>
    </head>
    <body style="background-color: moccasin; display: flex; flex-direction: column; align-items: center; text-align: justify; justify-content: center; min-height: 100vh; margin: 0; font-family: Arial, sans-serif;">
            <form id="analysisForm" action="" method="get" style="width: 100%; max-width: 600px; margin: 20px; text-align: center;">
                <label for="url"><strong>ENTER ARTICLE URL BELOW:</strong></label><br>
                <input type="url" id="url" name="url" value="{url}" style="width: 100%; padding: 10px; margin: 20px 0; strong;"><br>
                <input type="submit" value="ANALYSE ARTICLE" style="padding: 10px; background-color: #4CAF50; color: black; border: 3px black; cursor: pointer;">
            </form>

            <div id="loader" class="loader"></div>

            <div id="results" style="width: 100%; max-width: 800px; margin: 20px;">
                {extracted_article_data}
            </div>

            <button onclick="window.scrollTo({{ top: 0, behavior: 'smooth' }})" id="backToTopBtn" style="position: fixed; bottom: 20px; right: 20px; padding: 10px 15px; background-color: #555; color: white; border: none; border-radius: 4px; cursor: pointer; display: none;">Back to Top</button>
      
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)