import pandas as pd
import numpy as np
import scipy
from scipy.sparse.linalg import svds


def svd_pipeline():
    pass


def perform_svd(df_train):
    data_svd = df_train.copy()

    user_item_matrix = data_svd.pivot_table(index='customer_unique_id',
                                            columns='product_id',
                                            values='review_score',
                                            fill_value=0)
    print(user_item_matrix.shape)

    mask = np.isnan(user_item_matrix)
    masked_arr = np.ma.masked_array(user_item_matrix, mask)

    item_means = np.mean(masked_arr, axis=0)
    item_means_tiled = np.tile(item_means, (user_item_matrix.shape[0], 1))

    # Matrix Factorization
    U, sigma, V = svds(user_item_matrix, k=15)

    # Truncate sigma
    for i in range(5):
        sigma[i] = 0

    # Reconstruction
    sigma = np.diag([np.sqrt(sigma[i]) for i in range(0, 15)])

    Usk = np.dot(U, sigma)
    skV = np.dot(sigma, V)
    UsV = np.dot(Usk, skV)

    UsV = UsV + item_means_tiled

    predicted_df = pd.DataFrame(UsV)
    predicted_df.index = user_item_matrix.index
    predicted_df.columns = user_item_matrix.columns
    predicted_df["rec_product"] = predicted_df.idxmax(axis=1)
    predicted_df["customer"] = predicted_df.index

    return predicted_df


def extract_test_recommendations(df_test, predicted_df):
    test_users = df_test["customer_unique_id"].values.tolist()
    recommendations = predicted_df[predicted_df["customer"].isin(test_users)][["customer", "rec_product"]].copy()

    true_with_rec = df_test.merge(recommendations,
                                  left_on="customer_unique_id",
                                  right_on="customer")
    return true_with_rec
