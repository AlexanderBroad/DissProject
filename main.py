from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os
import nltk
import newspaper
import tldextract
from NewsSentiment import TargetSentimentClassifier
from nltk.tokenize import sent_tokenize
import re
import html

app = Flask(__name__)

def initialize_nltk():
    """Locate NLTK data"""
    
    # Add app directory to NLTK's data path
    nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))
 
def get_publication_details(url):
    """
    Extract publication details from the URL with error handling
    """
    try:
        # Use tldextract to get domain information
        ext = tldextract.extract(url)
        
        # Default values
        details = {
            "name": "Unknown Publication",
        }
        
        # Publication name mapping
        publication_mapping = {
            # Major global news publications
            "nytimes": "The New York Times",
            "washingtonpost": "The Washington Post",
            "wsj": "The Wall Street Journal",
            "guardian": "The Guardian",
            "bbc": "BBC News",
            "cnn": "CNN",
            "foxnews": "Fox News",
            "nbcnews": "NBC News",
            "cbsnews": "CBS News",
            "abcnews": "ABC News",
            "reuters": "Reuters",
            "apnews": "Associated Press",
            "bloomberg": "Bloomberg",
            "economist": "The Economist",
            "usatoday": "USA Today",
            "latimes": "Los Angeles Times",
            "chicagotribune": "Chicago Tribune",
            "huffpost": "HuffPost",
            "npr": "NPR",
            "forbes": "Forbes",
            "businessinsider": "Business Insider",
            "theatlantic": "The Atlantic",
            "politico": "Politico",
            "vox": "Vox",
            "slate": "Slate",
            
            # UK National Newspapers and News Sites
            "theguardian": "The Guardian",
            "thetimes": "The Times",
            "telegraph": "The Telegraph",
            "independent": "The Independent",
            "ft": "Financial Times",
            "dailymail": "Daily Mail",
            "mailonline": "Daily Mail",
            "mirror": "The Mirror",
            "express": "Daily Express",
            "thesun": "The Sun",
            "standard": "Evening Standard",
            "metro": "Metro",
            "dailystar": "Daily Star",
            "dailyrecord": "Daily Record",
            "observer": "The Observer",
            "ipaper": "i",
            "inews": "i News",
            "morningstar": "Morning Star",
            "newstatesman": "New Statesman",
            "spectator": "The Spectator",
            "economist": "The Economist",
            "private-eye": "Private Eye",
            "prospect": "Prospect Magazine",
            "standpoint": "Standpoint",
            
            # UK Online News Sites
            "unherd": "UnHerd",
            "bylinetimes": "Byline Times",
            "tortoise": "Tortoise Media",
            "theconversation": "The Conversation UK",
            "opendemocracy": "openDemocracy",
            "huffingtonpost": "HuffPost UK",
            "politicshome": "PoliticsHome",
            "politico": "POLITICO Europe",
            "politicaluk": "Political UK",
            "thecanary": "The Canary",
            "novara": "Novara Media",
            "tribunemag": "Tribune",
            "thejc": "The Jewish Chronicle",
            "pinknews": "PinkNews",
            "theduran": "The Duran",
            "thetab": "The Tab",
            
            # UK Regional/Local News
            "manchestereveningnews": "Manchester Evening News",
            "liverpoolecho": "Liverpool Echo",
            "birminghammail": "Birmingham Mail",
            "bristolpost": "Bristol Post",
            "leicestermercury": "Leicester Mercury",
            "nottinghampost": "Nottingham Post",
            "chroniclelive": "ChronicleLive",
            "walesonline": "WalesOnline",
            "leeds-live": "Leeds Live",
            "yorkshirepost": "The Yorkshire Post",
            "yorkshireeveningpost": "Yorkshire Evening Post",
            "thenational": "The National",
            "edinburghnews": "Edinburgh Evening News",
            "scotsman": "The Scotsman",
            "scottishsun": "The Scottish Sun",
            "glasgowlive": "Glasgow Live",
            "glasgowtimes": "Glasgow Times",
            "examiner": "Huddersfield Examiner",
            "coventrytelegraph": "Coventry Telegraph",
            "gazettelive": "Teesside Live",
            "bournemouthecho": "Bournemouth Echo",
            "plymouthherald": "Plymouth Herald",
            "shropshirestar": "Shropshire Star",
            "expressandstar": "Express & Star",
            "belfasttelegraph": "Belfast Telegraph",
            "irishnews": "The Irish News",
            "newsletter": "News Letter",
            "impartialreporter": "The Impartial Reporter",
            "derbyshiretimes": "Derbyshire Times",
            "kentonline": "Kent Online",
            "cambridge-news": "Cambridge News",
            "thenorthernecho": "The Northern Echo",
            "eveningtelegraph": "Evening Telegraph",
            "pressandjournal": "Press and Journal",
            "eveningexpress": "Evening Express",
            "southwalesargus": "South Wales Argus",
            "western-mail": "Western Mail",
            "eveningstandard": "Evening Standard",
            "theboltonnews": "The Bolton News",
            "thelancasterandmorecambecitizen": "The Lancaster and Morecambe Citizen",
            
            # UK Business/Finance
            "cityam": "City A.M.",
            "thisismoney": "This is Money",
            "investorschronicle": "Investors Chronicle",
            "moneyweek": "MoneyWeek",
            "business-live": "Business Live",
            "citywire": "Citywire",
            "ftadviser": "FT Adviser",
            "uktech": "UK Tech News",
            "siliconrepublic": "Silicon Republic",
            "techmarketview": "TechMarketView",
            
            # UK Magazines and Specialty Media
            "newscientist": "New Scientist",
            "thegrocer": "The Grocer",
            "farmersweekly": "Farmers Weekly",
            "nursing-times": "Nursing Times",
            "healthservicejournal": "Health Service Journal",
            "bmj": "The BMJ",
            "lrb": "London Review of Books",
            "timeshighereducation": "Times Higher Education",
            "thedrinksbusiness": "The Drinks Business",
            "architectsjournal": "Architects' Journal",
            "theengineer": "The Engineer",
            "computing": "Computing",
            "computerweekly": "Computer Weekly"
        }
        
        # Check for domains with and without 'www' prefix
        domain = ext.domain.lower()
        domain_with_suffix = f"{domain}.{ext.suffix}".lower()
        
        # Only update if we have valid domain information
        if domain:
            # First check if domain is in our mapping
            if domain in publication_mapping:
                details["name"] = publication_mapping[domain]
            # Then check domain with suffix
            elif domain_with_suffix in publication_mapping:
                details["name"] = publication_mapping[domain_with_suffix]
            # Check for common UK newspaper domains
            elif domain.startswith("the") and domain[3:] in publication_mapping:
                details["name"] = publication_mapping[domain[3:]]
            else:
                # Fallback to basic formatting
                if domain.startswith("the"):
                    # Properly capitalise "The" for publications
                    words = domain.split("-")
                    capitalized_words = [word.capitalize() for word in words]
                    details["name"] = " ".join(capitalized_words)
                else:
                    details["name"] = domain.replace("-", " ").title()
        
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

    # Extract individual words from publication name for better matching
    pub_name_words = set([word.lower() for word in pub_name_lower.split() if len(word) > 2])
    
    # Common publication words that might appear in author lists
    common_pub_words = [
        'www.facebook.com', 'news', 'times', 'post', 'daily', 'guardian', 'mail', 
        'journal', 'chronicle', 'tribune', 'gazette', 'herald', 'bbc', 'nyt', 'nytimes',
        'ap', 'reuters', 'associated press', 'press', 'media', 'network', 'agency',
        'magazine', 'publications', 'publisher', 'staff', 'correspondent', 'reporter',
        'editor', 'wire', 'syndicate', 'press association', 'bloomberg', 'cnbc', 'cnn'
    ]
    
    filtered_authors = []
    seen_authors = set()  # To track normalized versions of authors already added

    for author in authors:
        # Skip if the author name is empty or just whitespace
        if not author or author.strip() == '':
            continue
        
        author_lower = author.lower().strip()

        # Skip if the author name is the publication name
        if author_lower == pub_name_lower:
            continue
            
        # Skip if the author name contains the publication name and is less than 50 characters
        # (To avoid filtering out quotes that legitimately contain the publication name)
        if pub_name_lower in author_lower and len(author) < 50:
            # But don't skip if it looks like a genuine person with the publication
            if ',' not in author_lower and ' - ' not in author_lower:
                continue
            
        # Check if this author contains multiple words from the publication name
        author_words = set(author_lower.split())
        if len(pub_name_words.intersection(author_words)) >= min(2, len(pub_name_words)) and len(author) < 35:
            continue
            
        # Skip common publication identifiers
        if any(pub_word in author_lower for pub_word in common_pub_words) and len(author) < 25:
            continue
        
        # Create a normalized version of the author name
        normalized_author = ''.join(c.lower() for c in author if c.isalnum())
        
        # Skip duplicates
        if normalized_author in seen_authors:
            continue
            
        seen_authors.add(normalized_author)
        filtered_authors.append(author)
    
    return filtered_authors

def analyze_sentiment_newssentiment(text):
    """
    Analyze sentiment of text using NewsSentiment with sentence-level chunking
    for handling long texts and entity sentiment analysis
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
    <div style="margin-bottom: 15px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
        <strong>🔍 TOP ENTITIES:</strong>
        <label class="switch toggle-section" style="transform: scale(0.7);">
                <input type="checkbox" checked id="topEntitiesToggle">
                <span class="slider round"></span>
        </label>
        </div>
        <div class="collapsible-content active" id="topEntitiesContent">
    """
    
    for entity in top_entities:
        # Determine the color based on sentiment
        if entity["sentiment"] == "positive":
            border_colour = "#90EE90"  # Light green
            bg_colour = "#90EE90"
        elif entity["sentiment"] == "negative":
            border_colour = "#FFB6C1"  # Light pink
            bg_colour = "#FFB6C1" 
        else:
            border_colour = "#F0F8FF"  # Light blue
            bg_colour = "#A9D0F5"
        
        # Calculate confidence percentage for display
        confidence_pct = f"{entity['confidence'] * 100:.1f}%"
        confidence_width = f"{entity['confidence'] * 100}%"
        
        # Create the entity cards
        entities_html += f"""
        <div style="border: 1px solid #ddd; border-left: 5px solid {border_colour}; border-radius: 8px; 
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
                          background-color: {bg_colour};"></div>
            </div>
            <p style="text-align: right; font-size: 0.8em; margin: 5px 0 0 0;">Confidence: {confidence_pct}</p>
        </div>
        """
    
    entities_html += """
        </div>
    </div>
    """
    
    return entities_html

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
                <div id="sentiment-legend" style="opacity: 1; margin-bottom: 10px; transition: opacity 0.3s ease;">
                    <strong>📊 SENTIMENT LEGEND:</strong><br>
                    <span style="background-color: #90EE90; padding: 0 5px;">Positive</span>
                    <span style="background-color: #F0F8FF; padding: 0 5px; margin: 0 10px;">Neutral</span>
                    <span style="background-color: #FFB6C1; padding: 0 5px;">Negative</span>
                </div>
                
                <label class="switch">
                    <input type="checkbox" checked id="sentimentToggle">
                    <span class="slider round"></span>
                </label>
            </div>
            
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <strong>📋 SUMMARY:</strong>
                    <label class="switch toggle-section" style="transform: scale(0.7);">
                        <input type="checkbox" id="summaryToggle">
                        <span class="slider round"></span>
                    </label>
                </div>
                <div class="collapsible-content" id="summaryContent">
                    <div class="text-plain">{plain_summary}</div>
                    <div class="text-highlighted" style="display: none;">{highlighted_summary}</div>
                </div>
            </div>

            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <strong>📄 ARTICLE TEXT:</strong>
                    <label class="switch toggle-section" style="transform: scale(0.7);">
                        <input type="checkbox" checked id="articleToggle">
                        <span class="slider round"></span>
                    </label>
                </div>
                <div class="collapsible-content active" id="articleContent">
                    <div class="text-plain">{plain_text}</div>
                    <div class="text-highlighted" style="display: none;">{highlighted_text}</div>
                </div>
            </div>
        </div>

        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Toggle for summary section
            const summaryToggle = document.getElementById('summaryToggle');
            const summaryContent = document.getElementById('summaryContent');
            
            summaryToggle.addEventListener('change', function() {{
                if(this.checked) {{
                    summaryContent.classList.add('active');
                }} else {{
                    summaryContent.classList.remove('active');
                }}
            }});
            
            // Toggle for article text section
            const articleToggle = document.getElementById('articleToggle');
            const articleContent = document.getElementById('articleContent');
            
            articleToggle.addEventListener('change', function() {{
                if(this.checked) {{
                    articleContent.classList.add('active');
                }} else {{
                    articleContent.classList.remove('active');
                }}
            }});

            // Toggle for top entities section
            const topEntitiesToggle = document.getElementById('topEntitiesToggle');
            const topEntitiesContent = document.getElementById('topEntitiesContent');
            
            if(topEntitiesToggle && topEntitiesContent) {{
            topEntitiesToggle.addEventListener('change', function() {{
                if(this.checked) {{
                    topEntitiesContent.classList.add('active');
                }} else {{
                    topEntitiesContent.classList.remove('active');
                }}
            }}); 
        }}
        }});
        </script>
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
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=0.9">
        <title>Article Bias Indicator</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px;  }}

            /* Animation styles for collapsible sections */
            .collapsible-content {{
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.5s ease-out, opacity 0.3s ease-out;
                opacity: 0;
            }}
    
            .collapsible-content.active {{
                max-height: 10000px; /* Large enough to contain all content */
                opacity: 1;
                transition: max-height 0.5s ease-in, opacity 0.5s ease-in;
            }}

            /* Loading bar styles */
            .loader {{
                border: 16px solid #f3f3f3;
                border-radius: 50%;
                border-top: 16px solid #4CAF50;
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

            .fade-in {{
                transition: opacity 0.5s ease-in-out;
            }}

            .visible {{
                opacity: 1 !important;
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
                        const results = document.getElementById('results');
                        if (results) {{
                            setTimeout(function() {{
                                results.classList.add('visible');
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
                    const results = document.getElementById("results");
                    if (results) {{
                        results.classList.remove('visible');
                    }}
                    return true;
                }}
        </script>
    </head>
    <body style="background-color: navajowhite; display: flex; flex-direction: column; align-items: center; text-align: justify; justify-content: center; min-height: 100vh; margin: 0; font-family: Arial, sans-serif;">
            <div style="text-align: center; max-width: 600px; padding: 10px;"
            <p>This tool detects <b>sentiment</b> in news articles, indicating an author’s bias for or against a given subject. It can show positive, neutral, or negative sentiment towards people, locations, and organisations. <b>It will occasionally make mistakes</b> and is intended only as a starting point to get you thinking about author bias.<br>Created by Alexander Broad <a href="https://github.com/AlexanderBroad/DissProject" target="_blank">(GitHub repository)</a></p>
            </div>
            <form id="analysisForm" action="" method="get" style="width: 100%; max-width: 600px; margin: 20px; text-align: center; line-height: 1.6;">
                <label for="url"><strong>ENTER ARTICLE URL BELOW:</strong></label><br>
                <input type="url" id="url" name="url" value="{url}" style="width: 100%; padding: 10px; margin: 20px 0; strong;"><br>
                <input type="submit" value="ANALYSE ARTICLE" style="padding: 10px; background-color: #4CAF50; color: black; border: 3px black; cursor: pointer;">
            </form>

            <div id="loader" class="loader"></div>

            <div id="results" class="fade-in" style="width: 100%; max-width: 800px; margin: 20px; opacity: 0;">
                {extracted_article_data}
            </div>

            <button onclick="window.scrollTo({{ top: 0, behavior: 'smooth' }})" id="backToTopBtn" style="position: fixed; bottom: 20px; right: 20px; padding: 10px 15px; background-color: #555; color: white; border: none; border-radius: 4px; cursor: pointer; display: none;">Back to Top</button>
      
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(host='0.0.0.0')