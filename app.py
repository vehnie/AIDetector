from flask import Flask, render_template, request
import pickle
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, RegexpParser
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

with open('rf_model_1.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('svm_model_1.pkl', 'rb') as f:
    svm_model = pickle.load(f)


def format_text_with_features(model_selected,input_text):
    df = pd.DataFrame({'text': [input_text]})

    # remove special characters
    df['text'] = df['text'].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)

    # Lowercase the text
    df['text'] = df['text'].str.lower()
    
    # Remove extra spaces
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()

    
    # Tokenize the text
    df['tokens'] = df['text'].str.split()
    
    # Remove stop words
    df['tokens'] = df['text'].apply(word_tokenize)
    stop_words = set(stopwords.words('english'))
    df['tokens'] = df['tokens'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    df['lemmas'] = df['tokens'].apply(
    lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

    # text length and word count
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))


    content_tags = {'NN', 'VB', 'JJ', 'RB'}

    def calculate_lexical_density(lemmas):
        # POS tagging
        pos_tags = pos_tag(lemmas)
        
        # Filter lexical words
        lexical_words = [word for word, tag in pos_tags if tag[:2] in content_tags]
        
        # Calculate lexical density
        if len(lemmas) > 0:
            return len(lexical_words) / len(lemmas)
        else:
            return 0
    
    df['lexical_density'] = df['lemmas'].apply(calculate_lexical_density)


    def calculate_grammatical_complexity(text):
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)
        
        # Initialize variables for counting
        total_words = 0
        total_clauses = 0
        total_sentences = len(sentences)
        
        # For each sentence, tokenize words and count clauses
        for sentence in sentences:
            words = word_tokenize(sentence)
            total_words += len(words)
            
            # Count clauses based on the occurrence of coordinating conjunctions (e.g., "and", "but")
            # You can expand this list as needed
            conjunctions = ['and', 'but', 'or', 'nor', 'for', 'so', 'yet']
            clauses = sum([1 for word in words if word.lower() in conjunctions])
            
            total_clauses += clauses + 1  # Each sentence is considered at least 1 clause
        
        # Calculate average sentence length (in words)
        avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
        
        # Calculate the average number of clauses per sentence
        avg_clauses_per_sentence = total_clauses / total_sentences if total_sentences > 0 else 0
        
        # Combine metrics into a "grammatical complexity" score
        grammatical_complexity_score = avg_sentence_length * avg_clauses_per_sentence
        
        return grammatical_complexity_score

    df['grammatical_complexity'] = df['text'].apply(calculate_grammatical_complexity)

    # Function to calculate syntactic complexity with coordination, relative clauses, and subordination
    def calculate_syntactic_complexity(text):
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)
        
        # Initialize variables for counting
        total_tokens = 0
        total_phrases = 0  # Count of noun phrases, verb phrases, adjective phrases, etc.
        total_coordination = 0
        total_subordination = 0
        total_relative_clauses = 0
        total_sentences = len(sentences)
        
        # For each sentence, tokenize and perform POS tagging
        for sentence in sentences:
            words = word_tokenize(sentence)
            pos_tags = pos_tag(words)
            total_tokens += len(words)
            
            # Define a more comprehensive grammar to include noun phrases, verb phrases,
            # adjective phrases, prepositional phrases, coordination, relative clauses, and subordination
            grammar = r"""
                NP: {<DT>?<JJ>*<NN>}         # Noun Phrase
                VP: {<VB.*><NP|PP>*}          # Verb Phrase
                AP: {<JJ>}                    # Adjective Phrase
                PP: {<IN><NP>}                # Prepositional Phrase
                CC: {<CC>}                    # Coordinating Conjunction
                RC: {<WP|WDT><VB.*><NP>}      # Relative Clause (e.g., who, that, which)
                SBAR: {<IN><S>}               # Subordinating Clause (e.g., because, if)
            """
            
            # Create a parser based on the defined grammar
            cp = RegexpParser(grammar)
            tree = cp.parse(pos_tags)
            
            # Count the number of noun phrases, verb phrases, adjective phrases, prepositional phrases
            # as well as coordination, relative clauses, and subordination
            total_phrases += sum(1 for subtree in tree.subtrees() if subtree.label() in ['NP', 'VP', 'AP', 'PP'])

        
        # Calculate the average number of phrases per sentence
        avg_phrases_per_sentence = total_phrases / total_sentences if total_sentences > 0 else 0
        syntactic_complexity_score = total_tokens / avg_phrases_per_sentence if avg_phrases_per_sentence > 0 else 0
        
        return syntactic_complexity_score
    
    df['syntactic_complexity'] = df['text'].apply(calculate_syntactic_complexity).apply(pd.Series)


        # Define specific words related to AI-generated text
    specific_words = {
        'hedging': ["maybe", "perhaps", "possibly", "likely", "potentially", "could", "would", "might", "seem", "suggest", "appears", "likely", "suggests"],
        'filler': ["like", "you know", "actually", "basically", "seriously", "literally", "i mean", "so", "well"],
        'polite': ["could you", "please", "thank you", "sorry", "respectfully", "kindly"],
        'adverbs': ["definitely", "absolutely", "certainly", "undoubtedly", "clearly", "extremely", "completely", "fully"],
        'quantifiers': ["all", "every", "some", "many", "few", "most"],
        'ai_terms': ["algorithm", "model", "data", "training", "machine learning", "artificial intelligence", "neural network", "deep learning", "automation"]
    }

    # Define negation words
    negation_words = ["not", "no", "never", "none", "nothing", "nobody", "neither", "nowhere", "doesn't", "isn't", "wasn't", "aren't", "can't", "won't", "shouldn't", "haven't", "didn't"]

    # Define words for vague language
    vague_words = ["things", "stuff", "aspects", "elements", "factors", "issues"]

    # Define passive voice expressions (simplified for demonstration)
    passive_phrases = ["is done", "has been performed", "is considered"]

    # Function to count specific words and patterns
    def analyze_lemmas_for_specific_characteristics(lemmas):
        text = " ".join(lemmas).lower()  # Convert lemmas to lowercase string for easier matching
        
        # Count specific words related to AI-generated text
        hedging_count = sum(1 for word in lemmas if word in specific_words['hedging'])
        filler_count = sum(1 for word in lemmas if word in specific_words['filler'])
        polite_count = sum(1 for word in lemmas if word in specific_words['polite'])
        adverb_count = sum(1 for word in lemmas if word in specific_words['adverbs'])
        quantifier_count = sum(1 for word in lemmas if word in specific_words['quantifiers'])
        ai_terms_count = sum(1 for word in lemmas if word in specific_words['ai_terms'])
        
        # Count negation words
        negation_count = sum(1 for word in lemmas if word in negation_words)
        
        # Count vague words
        vague_word_count = sum(1 for word in lemmas if word in vague_words)
        
        # Count passive voice phrases (simplified check for combinations of words in passive_phrases)
        passive_count = sum(1 for phrase in passive_phrases if phrase in text)
        
        return {
            'hedging_word_count': hedging_count,
            'filler_word_count': filler_count,
            'polite_word_count': polite_count,
            'adverb_word_count': adverb_count,
            'quantifier_word_count': quantifier_count,
            'ai_term_count': ai_terms_count,
            'negation_word_count': negation_count,
            'vague_word_count': vague_word_count,
            'passive_voice_count': passive_count
        }
    
        # Example of applying this function to the 'lemmas' column
    df_features_lemmas = df['lemmas'].apply(analyze_lemmas_for_specific_characteristics)

    # Normalize the result and merge with the original dataframe
    df_features_lemmas = pd.json_normalize(df_features_lemmas)

    # Add the features back to the original dataframe
    df = pd.concat([df, df_features_lemmas], axis=1)

    # Function to calculate textual entropy
    def calculate_entropy(tokens):
        # Count the frequency of each token
        token_counts = Counter(tokens)
        
        # Total number of tokens
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return 0  # Return 0 if the list is empty
        
        # Calculate probabilities and entropy
        probabilities = [count / total_tokens for count in token_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)  # Avoid log(0)
        
        return entropy
    
    df['textual_entropy'] = df['tokens'].apply(calculate_entropy)

    df['is_ai'] = 0


    # Load the saved vectorizer
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    # Separate the target variable and features
    X = df.drop(columns=['is_ai'])
    y = df['is_ai']

    # Drop the 'tokens' and 'lemmas' columns as you don't want them
    X = X.drop(columns=['tokens', 'lemmas'])

    # Extract text column and other numerical columns
    X_text = X['text']  # Text column
    X_features = X.drop(columns=['text'])  # Other features

    # Transform the text data using the loaded vectorizer
    X_text_vec = vectorizer.transform(X_text)

    # Handle non-numeric columns in X_features
    X_numeric_features = X_features.select_dtypes(include=[np.number])

    # Scale the numeric features
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric_features)

    # Combine the vectorized text features and scaled numeric features
    X_combined = hstack((X_text_vec, X_numeric_scaled))

    # Create a new DataFrame with combined features and target variable
    df_processed = pd.DataFrame.sparse.from_spmatrix(X_combined)

    # Add back the non-text columns (keeping the ones you want)
    non_numeric_columns = X_features.select_dtypes(exclude=[np.number]).columns
    df_processed[non_numeric_columns] = X_features[non_numeric_columns]

    # Add the target variable back to the DataFrame
    df_processed['is_ai'] = y


    # drop is_ai column
    df_processed = df_processed.drop(columns=['is_ai'])

    if model_selected == 'random_forest':
        model= rf_model
    elif model_selected == 'svm':
        model = svm_model
    else:
        raise ValueError('Invalid model choice: {}'.format(model_selected))

    prediction = model.predict(df_processed)

    if prediction == 1:
        return "The text is AI-generated."
    else:
        return "The text is human-generated."

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        input_text = request.form['text']
        model_choice = request.form['model']
        # Call the format_text_with_features function with model_selected
        result = format_text_with_features(model_choice, input_text)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
