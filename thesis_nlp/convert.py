from gensim.models import Doc2Vec
import numpy

doc2vec_model_location = 'model/doc2vec-model.bin'

class NewsConverter:

    def __init__(self):
        #self.model = Doc2Vec.load(doc2vec_model_location)
        print("Init NewsConverter")

    def convert_doc_to_vector(self, doc):
        #return self.model.infer_vector(doc.split())
        return numpy.zeros(300)

