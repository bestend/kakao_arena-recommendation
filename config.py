import os
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = ROOT_DIR + '/../res/'
CACHE_DIR = ROOT_DIR + '/../cache/'

MAX_USER_SEQUENCE_LEN = 20
MAX_SEARCH_KEYWORD_LEN = 20
ARTICLE_EMBEDDING_SIZE = 200

BEGIN_ARTICLE_TIME = datetime(2015, 1, 1, 0, 0)
END_ARTICLE_TIME = datetime(2019, 2, 22, 0, 0)

TOKEN_PAD = '<PAD>'  # Token for padding
TOKEN_UNK = '<UNK>'  # Token for unknown words
VALUE_PAD = 0
VALUE_UNK = 1
