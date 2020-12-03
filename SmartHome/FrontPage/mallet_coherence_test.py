from gensim.models import CoherenceModel, LdaModel
from gensim.corpora.dictionary import Dictionary

def main():
    texts = [
        ['human', 'interface', 'computer'],
        ['survey', 'user', 'computer', 'system', 'response', 'time'],
        ['eps', 'user', 'interface', 'system'],
        ['system', 'human', 'system', 'eps'],
        ['user', 'response', 'time'],
        ['trees'],
        ['graph', 'trees'],
        ['graph', 'minors', 'trees'],
        ['graph', 'minors', 'survey']
    ]

    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    goodLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=50, num_topics=2)
    badLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=1, num_topics=2)

    goodcm = CoherenceModel(model=goodLdaModel, texts=texts, corpus=corpus, dictionary=dictionary, coherence='c_v')
    badcm = CoherenceModel(model=badLdaModel, corpus=corpus, dictionary=dictionary, coherence='u_mass')

    print(badcm.get_coherence())
    print(goodcm.get_coherence())

if __name__ == "__main__":
    main()
