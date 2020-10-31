import pandas as pd


def delete_duplicated_recommendations(data):
    data_deduplicated = data.drop_duplicates(subset=["product_id", "customer_unique_id", "rec_product"], keep="first")
    return data_deduplicated


def assess_similarity(data, similarities):

    def extract_similarity(product1,product2, similarity_matrix):
        similarity = similarity_matrix[similarity_matrix["product_id"] == product1][product2].values[0]
        return similarity

    data["similarity"] = data.apply(lambda x: extract_similarity(x['product_id'], x['rec_product'],
                                                                 similarities), axis=1)

    return data
