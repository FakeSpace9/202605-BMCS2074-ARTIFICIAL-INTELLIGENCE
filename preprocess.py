"""
Text preprocessing pipeline for IT_Support_Ticket_Data.csv
Cleans the 'Body' column so it's ready for TF-IDF / model training.

Steps: lowercase -> remove PII placeholders -> remove punctuation
       -> tokenize -> remove stopwords -> lemmatize
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Run once to download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Optional: extra domain-specific "filler" words that appear in almost
# every ticket regardless of department (greetings/sign-offs), so they
# add noise rather than signal. Add/remove words here as you see fit.
CUSTOM_STOPWORDS = {
    'dear', 'customer', 'support', 'team', 'regards', 'sincerely',
    'thank', 'thanks', 'please', 'hi', 'hello'
}


def clean_text(text, remove_custom_stopwords=False):
    if not isinstance(text, str):
        return ""

    # 1. Lowercase everything
    text = text.lower()

    # 2. Remove placeholder / personal identifiers
    #    This dataset anonymises names, account numbers, and phone numbers
    #    using tokens like "[Your Name]", "name", "acc_num", "tel_num"
    #    (sometimes squashed together, e.g. "nameacc_numtel_num").
    text = re.sub(r'\[your name\]', ' ', text)
    text = re.sub(r'\bacc_num\b', ' ', text)
    text = re.sub(r'\btel_num\b', ' ', text)
    text = re.sub(r'\bname\b', ' ', text)

    # Also strip real emails / phone numbers / long digit sequences,
    # in case any slipped through un-anonymised.
    text = re.sub(r'\S+@\S+', ' ', text)                   # emails
    text = re.sub(r'\+?\d[\d\-\s]{7,}\d', ' ', text)        # phone-like numbers
    text = re.sub(r'\b\d{4,}\b', ' ', text)                 # long standalone digit strings

    # 3. Remove special characters / punctuation (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 4. Collapse extra whitespace left behind by the steps above
    text = re.sub(r'\s+', ' ', text).strip()

    # 5. Tokenize
    tokens = word_tokenize(text)

    # 6. Remove stopwords (and any leftover 1-letter tokens)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    if remove_custom_stopwords:
        tokens = [t for t in tokens if t not in CUSTOM_STOPWORDS]

    # 7. Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)


if __name__ == "__main__":
    # Update this path to wherever your CSV is saved locally
    df = pd.read_csv('IT_Support_Ticket_Data.csv')

    # Drop the row(s) with missing Body text
    df = df.dropna(subset=['Body']).reset_index(drop=True)

    # Apply cleaning to every ticket
    df['clean_text'] = df['Body'].apply(clean_text)

    # Quick sanity check
    print(df[['Body', 'clean_text']].head(3).to_string())
    print("\nRows after cleaning:", len(df))

    # Save the cleaned dataset for the next step (TF-IDF / model training)
    df.to_csv('cleaned_tickets.csv', index=False)
    print("Saved cleaned_tickets.csv")