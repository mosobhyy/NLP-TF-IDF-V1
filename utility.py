import math
import re


def text_preprocessing(text):
    # Setting every word to lower
    text = text.lower()

    # Removing punctuations
    text = re.sub(r'[()\[\]{}!-/–;:\'",<>./?@#$%^&*_“~\\]', ' ', text)

    # Removing digits
    text = re.sub(r'\d', '', text)

    # Removing sequential whitespaces
    text = re.sub(r'\s+', ' ', text)

    # Removing leading and trailing whitespaces
    text = text.strip()

    return text


def tokenization(text):
    text = text.split()
    text = list(set(text))
    text.sort()

    return text


def create_domain_count(text, domain) -> dict:
    domain_count = {}
    for word in text:
        # Split used to avoid count every character in a word   (EX: avoid count 'a' in great)
        split_sentence = domain.split()
        domain_count[word] = split_sentence.count(word)

    return domain_count


def create_domain_tf(domain_count) -> dict:
    total = sum(domain_count.values())
    domain_tf = {}
    for word, count in domain_count.items():
        domain_tf[word] = count / total

    return domain_tf


def create_idf(domain1_count, domain2_count, num_of_domains=2):
    idf = {}

    for (word, count1), (count2) in zip(domain1_count.items(), domain2_count.values()):
        # Calculate how many docs has the word
        total_docs_count = 0
        total_docs_count += 1 if count1 > 0 else 0
        total_docs_count += 1 if count2 > 0 else 0

        idf[word] = math.log(num_of_domains / total_docs_count, 10)

    return idf


def create_tf_idf(domain1_tf, idf):
    tf_idf = {}

    for (word, tf), (idf_value) in zip(domain1_tf.items(), idf.values()):
        tf_idf[word] = tf * idf_value

    return tf_idf
