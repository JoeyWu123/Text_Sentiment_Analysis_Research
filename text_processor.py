import re,string
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
# nltk.download('stopwords')
import nltk

#the following part refers to https://craigrshenton.github.io/home/data-smarts-7/
negative_words=['not','no','don', "don't",'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
                'wouldn', "wouldn't","can't","cannot","cant","doesnt","dont","arent","isnt","wasnt","werent",
                "didnt","hadnt","havent","mustnt","shouldnt","wouldnt"]  #come from stopwords from nltk, plus some add by me
punctuation = list(string.punctuation)
stop_words = stopwords.words('english') + punctuation + ["i'm","that's","they're",
                                                         'rt', 'via', '…', '¿', '“', '”']
regex_str = [
    r'<[^>]+>',
    r'(?:@[\w_]+)',
    r"(?:\#+[\w_]+[\w_\-]*[\w_]+)",
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',
    r"(?:[a-z][a-z'\-_]+[a-z])",
    r'(?:[\w_]+)',
]
token_pattern = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def preprocess_tweet(tweet):
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with space
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', '', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    return tweet



# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def Lemmatizer(s):
    tagged_sent = pos_tag(s)  # 获取单词词性
    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
    return lemmas_sent

def removeStopWord_except_negative_words(str):
    result =[]
    for s in str:
        if(s.lower() in negative_words):
            result.append('not')
        elif(s.lower() in stop_words):
            continue
        else:
            result.append(s.lower())
    return result

def tokenize(tokens_re, s):
    return tokens_re.findall(s)

def processText_for_sentiment_analysis(tweet):

    tweet = preprocess_tweet(tweet)
    words = tokenize(token_pattern, tweet)  # cut sentence to list
    words = removeStopWord_except_negative_words(words)
    words = Lemmatizer(words)
    # print("s2: ", len(sentense), sentense)
    return words
