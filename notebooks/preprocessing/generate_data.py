import datetime
import random
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from preprocessing.lda_preprocess import lda_preprocess_pipeline
from preprocessing.lda import lda_pipeline
import pandas as pd
import numpy as np
import scipy.stats as stats


# Join datasets
def join_datasets(dataset1, dataset2, key):
    dataset1[key] = dataset1[key].astype("str")
    dataset2[key] = dataset2[key].astype("str")

    joined_datasets = dataset1.merge(dataset2, on=key, how="left")
    return joined_datasets


def join_all_datasets(
        data_orders, data_orders_items, data_clients, data_products, data_reviews
):
    data_orders_customers = join_datasets(data_orders, data_clients, key="customer_id")

    data_orders_items_customers = join_datasets(
        data_orders_items, data_orders_customers, key="order_id"
    )

    data_orders_items_customers_reviews = join_datasets(
        data_orders_items_customers, data_reviews, key="order_id"
    )

    data_orders_items_customers_reviews_products = join_datasets(
        data_orders_items_customers_reviews, data_products, key="product_id"
    )

    return data_orders_items_customers_reviews_products


def extract_data_subset_review(
        data, correlations, text_column_name, category_column_name
):
    data[text_column_name] = data[text_column_name].fillna(data["review_comment_title"])
    data = data.replace({'product_category_name': 'eletrodomesticos_2'},
                        {'product_category_name': 'eletrodomesticos'}, regex=True)

    prod_more_100_rev = data[data[text_column_name].notnull()][
        "product_id"
    ].value_counts()
    products_more_100 = prod_more_100_rev[prod_more_100_rev > 50].index.tolist()

    data_filtered = data[data[text_column_name].notnull() & data["product_id"].isin(products_more_100)].copy()

    more_100_rev = data_filtered[data_filtered[text_column_name].notnull()][
        category_column_name
    ].value_counts()
    categories_more_100 = more_100_rev[more_100_rev > 100].index.tolist()

    data_filtered_again = data_filtered[
        data_filtered[category_column_name].isin(categories_more_100)].copy()

    correlations_filtered = correlations[
        correlations["category"].isin(categories_more_100)
    ].copy()
    return data_filtered_again, correlations_filtered


def extract_products(data):
    return data["product_id"].unique().tolist()


def generate_num_products(miu=20, sigma=5):
    return int(np.random.normal(miu, sigma))


def generate_category_strategy(segment):
    alpha_dict = {1: [0.2, 0.8], 2: [0.8, 0.2], 3: [0.5, 0.5]}
    return np.random.dirichlet(alpha_dict[segment], 1).transpose()


def random_date():
    """Generate a random datetime between `start` and `end`"""
    start = datetime.date(2016, 1, 1)
    end = datetime.date(2018, 12, 1)
    random_date_generated = start + datetime.timedelta(
        # Get a random amount of seconds between `start` and `end`
        seconds=random.randint(0, int((end - start).total_seconds())),
    )
    return random_date_generated.strftime("%Y-%m-%d")


def generate_products(user, data, categories_correlations, similarities):
    n_products = generate_num_products()
    last_purchase = (
        data[data["customer_unique_id"] == user]
            .sort_values(by="order_purchase_timestamp", ascending=False)
            .head(1)
    )
    last_user_product = last_purchase["product_id"].values[0]
    last_price = last_purchase["price"].values[0]
    last_product_category = last_purchase["product_category_name"].values[0]

    category_prices = data[data["product_category_name"] == last_product_category][
        "price"
    ].values
    quantile = stats.percentileofscore(category_prices, last_price, "mean") / 100
    client_segment = np.random.choice([1, 2, 3])
    user_dirichlet = generate_category_strategy(client_segment)

    product_full_df = pd.DataFrame()

    for i in range(int(n_products * 0.7)):
        product_strategy = np.random.choice(
            ["similar", "complementary"], p=user_dirichlet.ravel().tolist()
        )
        if product_strategy == "complementary":
            next_category = np.random.choice(
                categories_correlations[
                    categories_correlations[last_product_category] == 1
                    ]["category"].values.tolist()
            )

            next_product_price_quantile_value = data[
                data["product_category_name"] == next_category
                ]["price"].quantile(quantile)
            cv = np.random.normal(0.3, 0.1)
            next_product_price_min = (
                    next_product_price_quantile_value
                    - next_product_price_quantile_value * cv
            )
            next_product_price_max = (
                    next_product_price_quantile_value
                    + next_product_price_quantile_value * cv
            )
            products_subset = data[
                (data["product_category_name"] == next_category)
                & (data["price"] > next_product_price_min)
                & (data["price"] < next_product_price_max)
                & (data["review_score"] >= 4)
                ]["product_id"].values

            if len(products_subset) == 0:
                products_subset = data[
                    data["product_category_name"] == next_category]["product_id"].values

            next_product = np.random.choice(products_subset)
            next_price = data[data["product_id"] == next_product]["price"].values[0]

        else:
            next_product = similarities[similarities[last_user_product] != 1][
                last_user_product
            ].argmax()

            next_price = data[data["product_id"] == next_product]["price"].values[0]
            next_category = data[data["product_id"] == next_product][
                "product_category_name"
            ].values[0]

        next_timestamp = random_date()
        next_review = np.random.choice([4, 5])

        product_df = pd.DataFrame(
            {
                "product_category_name": next_category,
                "product_id": next_product,
                "price": next_price,
                "order_purchase_timestamp": next_timestamp,
                "review_score": next_review
            },
            index=[0],
        )
        product_full_df = pd.concat([product_full_df, product_df], axis=0, sort=False)

        product_full_df.reset_index(inplace=True, drop=True)

    for i in range(int(n_products * 0.3)):
        next_product = np.random.choice(data["product_id"].values.tolist())
        next_price = data[data["product_id"] == next_product]["price"].values[0]
        next_category = data[data["product_id"] == next_product][
            "product_category_name"
        ].values[0]
        next_timestamp = random_date()
        next_review = np.random.choice([1, 2, 2, 2, 3])

        product_df = pd.DataFrame(
            {
                "product_category_name": next_category,
                "product_id": next_product,
                "price": next_price,
                "order_purchase_timestamp": next_timestamp,
                "review_score": next_review
            },
            index=[0],
        )
        product_full_df = pd.concat([product_full_df, product_df], axis=0, sort=False)

        product_full_df.reset_index(inplace=True, drop=True)

    product_full_df["customer_unique_id"] = user
    product_full_df["segment"] = client_segment

    return product_full_df


def generate_data(data_filtered, corelations_filtered, similarities, customers_num=-1):
    customers = data_filtered["customer_unique_id"].unique().tolist()

    all_customers = pd.DataFrame()

    for customer in customers[:customers_num]:
        customer_product_full_df = generate_products(customer, data_filtered, corelations_filtered, similarities)
        all_customers = pd.concat([all_customers, customer_product_full_df])

    return all_customers


# Extract Similarities
def extract_latent_components(
        data_joined, text_column_name, category_column_name, product_id_column, num_topics
):
    more_100_rev = data_joined[data_joined[text_column_name].notnull()][
        category_column_name
    ].value_counts()

    categories_more_100 = more_100_rev[more_100_rev > 100].index.tolist()

    data_for_lda_all = data_joined[
        data_joined[category_column_name].isin(categories_more_100)
        & data_joined[text_column_name].notnull()
        ]

    all_assignements = pd.DataFrame()

    for category in categories_more_100:
        data_for_lda = data_for_lda_all[
            data_for_lda_all[category_column_name] == category
            ].reset_index(drop=True)
        processed_docs = lda_preprocess_pipeline(data_for_lda, text_column_name)

        category_df = lda_pipeline(
            data_for_lda, processed_docs, product_id_column, num_topics
        )

        all_assignements = pd.concat([all_assignements, category_df])

    aggregated_product_view = all_assignements.groupby(all_assignements.index).mean()

    return aggregated_product_view


def extract_category_encoding(
        data_joined, text_column_name, category_column_name, product_id_column, num_topics
):
    more_100_rev = data_joined[data_joined[text_column_name].notnull()][
        category_column_name
    ].value_counts()

    categories_more_100 = more_100_rev[more_100_rev > 100].index.tolist()

    data_for_lda = data_joined[
        data_joined[category_column_name].isin(categories_more_100)
        & data_joined[text_column_name].notnull()
        ]

    data_for_lda = data_for_lda.reset_index(drop=True)

    processed_docs = lda_preprocess_pipeline(data_for_lda, text_column_name)

    category_df_latent = lda_pipeline(
        data_for_lda, processed_docs, product_id_column, num_topics
    )

    return category_df_latent


# Cast to datetime
def cast_columns(data, dt_columns, str_columns, num_columns):
    return data


def calculate_distances(aggregated_product_view):
    distances = pd.DataFrame(
        cosine_similarity(
            aggregated_product_view.values, aggregated_product_view.values
        )
    )
    distances.columns = aggregated_product_view.index
    distances.index = aggregated_product_view.index

    return distances


def enrich_generated_data():
    pass


def aggregate_category_view(data_filtered, category_df_latent):
    category_df_latent["product_id"] = category_df_latent.index
    merged_latent_category = category_df_latent.merge(data_filtered[["product_id",
                                                                     "product_category_name"]],
                                                      left_on="product_id",
                                                      right_on="product_id",
                                                      how="left")

    merged_latent_category_agg = merged_latent_category.groupby("product_category_name").mean().reset_index()
    return merged_latent_category_agg


def create_product_characteristics_complete(data_filtered, aggregated_product_view, merged_latent_category_agg):
    aggregated_product_view["product_id"] = aggregated_product_view.index
    columns_product_related = ["product_id",
                               "product_name_lenght",
                               "product_description_lenght",
                               "product_weight_g",
                               "product_photos_qty",
                               "product_length_cm",
                               "product_width_cm",
                               "product_height_cm",
                               "product_category_name"]

    columns_product = ["product_id",
                       "price",
                       "product_name_lenght",
                       "product_description_lenght",
                       "product_weight_g",
                       "product_photos_qty",
                       "product_length_cm",
                       "product_width_cm",
                       "product_height_cm",
                       "product_category_name"]

    data_filtered_unique = data_filtered.drop_duplicates(subset=columns_product_related)

    product_characteristics = aggregated_product_view.merge(data_filtered_unique[columns_product],
                                                            left_on="product_id",
                                                            right_on="product_id",
                                                            how="left")

    product_characteristics_category = product_characteristics.merge(merged_latent_category_agg,
                                                                     left_on="product_category_name",
                                                                     right_on="product_category_name",
                                                                     how="left")
    product_characteristics_category.drop(columns=["product_category_name"], inplace=True)

    product_characteristics_category.set_index("product_id", inplace=True)
    return product_characteristics_category
