import nltk
from nltk.tag.stanford import StanfordNERTagger
from newspaper import Article
from .utils import remove_non_ascii


class Extractor(object):
    def __init__(self, text=None, url=None):
        if not text and not url:
            raise Exception('text or url is required')

        self.text = text
        self.url = url
        self.places = []
        self.people = []
        self.organs = []
    
    def set_text(self):
        if not self.text and self.url:
            a = Article(self.url)
            a.download()
            a.parse()
            self.text = a.text


    def find_entities(self):
        self.set_text()

        text = nltk.word_tokenize(self.text)
        st = StanfordNERTagger('../ner-model.ser.gz', '../stanford-ner.jar')
        nes = nltk.ne_chunk(st.tag(text))

        for ne in nes:
            if type(ne) is nltk.tree.Tree:
                if ne.label() == 'GPE':
                    self.places.append(u' '.join([i[0] for i in ne.leaves()]))
                elif ne.label() == 'PERSON':
                    self.people.append(u' '.join([i[0] for i in ne.leaves()]))
                elif ne.label() == 'ORGANIZATION':
                    self.organs.append(u' '.join([i[0] for i in ne.leaves()]))
