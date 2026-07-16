"""
Text preprocessing pipeline for IT_Support_Ticket_Data.csv
Cleans 'Body' + folds in 'Tags' (topic keywords) so downstream
models get much stronger topical signal.
Steps: lowercase -> remove PII placeholders -> remove punctuation
-> tokenize -> remove stopwords -> lemmatize
"""

import re
import ast
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

CUSTOM_STOPWORDS = {
    # Greetings, sign-offs, and generic support-email boilerplate.
    'dear', 'customer', 'support', 'team', 'regards', 'sincerely',
    'thank', 'thanks', 'please', 'hi', 'hello',

    # High-frequency wording that describes the act of writing an email, not
    # the ticket's issue, routing queue, or urgency.  Keep words such as
    # urgent, critical, outage, error, unable, and down because they carry
    # useful priority or routing information.
    'hope', 'message', 'find', 'well', 'reaching', 'regarding', 'facing',
    'earliest', 'could', 'would', 'ensure', 'writing', 'kindly',
    'appreciate', 'provide', 'assistance', 'guidance', 'information',
    'request', 'look', 'forward', 'greatly', 'might', 'may', 'due',
    'recent', 'still', 'soon', 'possible', 'need', 'help', 'also', 'able',
    'like', 'want', 'br', 'u'
}

def clean_text(text, remove_custom_stopwords=False):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # FIX 1: Removed \b so it catches squashed placeholders (e.g. nameacc_numtel_num)
    text = re.sub(r'\[your name\]', ' ', text)
    text = re.sub(r'acc_num', ' ', text)
    text = re.sub(r'tel_num', ' ', text)
    text = re.sub(r'name', ' ', text)

    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\+?\d[\d\-\s]{7,}\d', ' ', text)
    text = re.sub(r'\b\d{4,}\b', ' ', text)

    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    if remove_custom_stopwords:
        tokens = [t for t in tokens if t not in CUSTOM_STOPWORDS]

    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


def parse_tags(tags_str):
    """Tags column is stored as a string like "['Account', 'Outage']".
    Parse it safely and join into a lowercase space-separated string
    so it can be appended to clean_text as extra vocabulary."""
    if not isinstance(tags_str, str) or not tags_str.strip():
        return ""
    try:
        tags_list = ast.literal_eval(tags_str)
        if isinstance(tags_list, list):
            # Repeat each tag word so it carries more TF-IDF weight
            # than a single mention buried in a long body of text
            return ' '.join([str(t).lower() for t in tags_list] * 2)
    except (ValueError, SyntaxError):
        pass
    return ""


if __name__ == "__main__":
    df = pd.read_csv('IT_Support_Ticket_Data.csv')
    df = df.dropna(subset=['Body']).reset_index(drop=True)

    # ---------------------------------------------------------
    # Department consolidation for the routing task
    # ---------------------------------------------------------
    # The original data has several overlapping support departments.  For a
    # first-line routing model, use three operationally meaningful queues:
    # Technical Operations, Customer and Product Services, and Billing and
    # Returns.  Human Resources is removed because it is an internal service
    # rather than a customer-support routing destination.
    #
    # Important: report this as a *three-class routing task*.  Do not describe
    # the resulting accuracy as performance on the original ten departments.
    df = df[df['Department'] != 'Human Resources'].reset_index(drop=True)

    department_mapping = {
        'Technical Support': 'Technical Operations',
        'IT Support': 'Technical Operations',
        'Service Outages and Maintenance': 'Technical Operations',
        'Customer Service': 'Customer and Product Services',
        'Product Support': 'Customer and Product Services',
        'Sales and Pre-Sales': 'Customer and Product Services',
        'General Inquiry': 'Customer and Product Services',
        'Billing and Payments': 'Billing and Returns',
        'Returns and Exchanges': 'Billing and Returns',
    }
    df['Department'] = df['Department'].replace(department_mapping)

    # Fail early if a source label was not deliberately handled above.
    expected_departments = {
        'Technical Operations',
        'Customer and Product Services',
        'Billing and Returns',
    }
    unexpected = set(df['Department'].unique()) - expected_departments
    if unexpected:
        raise ValueError(f'Unexpected Department values after mapping: {unexpected}')
    
    # ---------------------------------------------------------

    # FIX 2: Added lambda to actually trigger remove_custom_stopwords=True
    df['clean_text'] = df['Body'].apply(lambda x: clean_text(x, remove_custom_stopwords=True))

    # Fold in Tags as extra signal, if the column exists
    if 'Tags' in df.columns:
        df['clean_tags'] = df['Tags'].apply(parse_tags)
        df['clean_text'] = (df['clean_text'] + ' ' + df['clean_tags']).str.strip()
        df = df.drop(columns=['clean_tags'])

    # Print a sample and the final class distribution for the report.
    print(df[['Department', 'Body', 'clean_text']].head(3).to_string())
    print("\nRows after cleaning:", len(df))
    print("\nFinal routing-class distribution:\n", df['Department'].value_counts())

    df.to_csv('cleaned_tickets.csv', index=False)
    print("Saved cleaned_tickets.csv")
