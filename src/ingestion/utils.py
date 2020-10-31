import datetime
import random

def extract_alpha_categories(data_joined):
    revenues_groups = data_joined.groupby(by="product_category_name").agg({"price": "sum"})
    categories_products = revenues_groups.index.tolist()
    alpha_dir = tuple(revenues_groups.price.tolist())

    return alpha_dir, categories_products


def random_date():
    """Generate a random datetime between `start` and `end`"""
    start = datetime.date(2016, 1, 1)
    end = datetime.date(2018, 12, 1)

    return start + datetime.timedelta(
        # Get a random amount of seconds between `start` and `end`
        seconds=random.randint(0, int((end - start).total_seconds())),
    )
