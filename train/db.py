import urllib.parse
from sqlalchemy import create_engine
import pandas as pd

def get_engine(user, password, host, port, database):
    """
    返回 SQLAlchemy engine
    必须进行 URL 编码，否则密码内含 @ : / 会报错
    """
    pwd = urllib.parse.quote_plus(password)

    url = (
        f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{database}"
    )

    return create_engine(url)


def load_traces(engine):
    sql = """
        SELECT transaction_hash, trace
        FROM token_3crv_traces
    """
    return pd.read_sql(sql, engine)
