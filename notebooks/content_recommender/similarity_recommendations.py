

def recommend_similar(data, similarities):

    similarities_copy = similarities.copy()
    similarities_copy.index = similarities_copy["product_id"]

    def extract_similar(product1, similarity_matrix):
        rec_product = similarity_matrix[similarity_matrix[product1] != 1][product1].argmax()
        return rec_product

    data["rec_product"] = data.apply(lambda x: extract_similar(x['product_id'],
                                                               similarities_copy), axis=1)
    return data


def sim_rec_pipeline(df_train, similarities):
    df_train_with_rec = recommend_similar(df_train, similarities)
    return df_train_with_rec


def merge_sim_rec_with_test(df_train_with_rec,df_test ):
    df_train_with_rec_filtered = df_train_with_rec[df_train_with_rec["review_score"]>3].copy()

    sim_rec_test = df_test.merge(df_train_with_rec_filtered[["customer_unique_id", "rec_product"]])
    return sim_rec_test


def delete_duplicated_recommendations(data):
    data_deduplicated = data.drop_duplicates(subset=["product_id", "customer_unique_id", "rec_product"], keep="first").reset_index(drop=True)
    return data_deduplicated





