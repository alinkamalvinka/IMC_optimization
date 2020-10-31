import gensim
import numpy as np
import pandas as pd
from gensim import corpora, models
from gensim.models import ldaseqmodel
from gensim.models.coherencemodel import CoherenceModel

import pickle

np.random.seed(17)


def corpus_creation(processed_docs):
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')

    return corpus, dictionary


def model_lda(num_topics, corpus_tfidf, dictionary_lda):
    lda_model = gensim.models.ldamodel.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary_lda, passes=15)
    lda_model.save('model5.gensim')
    # topics = lda_model.print_topics(num_words=10)
    # for topic in topics:
    #     print(topic)


def tf_idf(corpus_tfidf):
    tfidf = models.TfidfModel(corpus_tfidf)
    corpus_lda = tfidf[corpus_tfidf]
    return corpus_lda


def model_lda_tfidf(num_topics, corpus_lda, dictionary_lda):
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_lda, num_topics=num_topics, id2word=dictionary_lda, passes=2,
                                                 workers=4)
    topics = lda_model_tfidf.print_topics(num_words=10)
    # for topic in topics:
    #     print(topic)

    return lda_model_tfidf


def dtm_lda(corpus, dictionary, time_slice, num_topics):
    dtm = ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=dictionary, time_slice=time_slice, num_topics=num_topics)
    return dtm


def extract_time_slices(df, time_var):
    time_intervals, time_slices = np.unique(df[time_var], return_counts=True)
    return time_intervals, time_slices


def assign_topic(lda_model_tfidf, corpus_lda):
    doc_lda = lda_model_tfidf[corpus_lda]
    return doc_lda


def get_coherence(lda_model, corpus, dictionary, texts):
    perplexity = lda_model.log_perplexity(corpus)
    print('Perplexity: ', perplexity)

    coherence_lda = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_u = coherence_lda.get_coherence()
    print('Coherence Score U-mass: ', coherence_u)

    coherence_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_c = coherence_lda.get_coherence()
    print('Coherence Score C_V: ', coherence_c)

    return perplexity, coherence_u, coherence_c


def lda_pipeline(pre_processed_docs, processed_docs, product_id_column, num_topics):
    corpus, dictionary = corpus_creation(processed_docs)
    corpus_tf_idf_data = tf_idf(corpus)
    lda_model_tfidf = model_lda_tfidf(num_topics, corpus_tf_idf_data, dictionary)
    prod_ids = pre_processed_docs[product_id_column].values

    products_df = pd.DataFrame()
    for i, row in enumerate(lda_model_tfidf[corpus_tf_idf_data]):
        percentage = []
        product_id = prod_ids[i]
        # Get the Dominant topic, Percent Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            percentage.append(prop_topic)

        product_df = pd.DataFrame({product_id: percentage})
        product_df_trans = product_df.transpose()
        products_df = pd.concat([products_df, product_df_trans])

    return products_df


