import os
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv

# =====================================
# 1. 加载环境变量
# =====================================
load_dotenv()

# =====================================
# 2. 本地 CSV
# =====================================
LOCAL_CSV = "function_sequence.csv"#后面可以一张一张表来join，可以放入数据库，现在trace还没放入数据库，先放本地来测试
df_local = pd.read_csv(LOCAL_CSV)
print(f"本地 CSV 加载成功，行数: {len(df_local)}")

# =====================================
# 3. 数据库配置（从 .env 读取）
# =====================================
DB_USER = os.getenv("POSTGRESQL_USER")
DB_PASS_RAW = os.getenv("POSTGRESQL_PASSWORD")
DB_HOST = os.getenv("POSTGRESQL_HOST")
DB_PORT = os.getenv("POSTGRESQL_PORT")
DB_NAME = os.getenv("POSTGRESQL_DATABASE")

if DB_PASS_RAW is None:
    raise RuntimeError("POSTGRESQL_PASSWORD 未从 .env 读到，请检查路径或文件名")

# URL 编码密码（处理 @ 等特殊字符）
DB_PASS = quote_plus(DB_PASS_RAW)

# 创建 engine
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("SQLAlchemy engine 创建成功")

# =====================================
# 4. 从远程读取 crv_transactions
# =====================================
df_remote = pd.read_sql("SELECT * FROM crv_transactions;", engine)
print(f"远程 crv_transactions 表读取成功，行数: {len(df_remote)}")

# =====================================
# 5. LEFT JOIN
# =====================================
df_joined = df_local.merge(
    df_remote,
    on="transaction_hash",
    how="left"
)
print(f"JOIN 完成，共 {len(df_joined)} 行")

# =====================================
# 6. 保存到本地
# =====================================
OUTPUT = "joined_results.csv"
df_joined.to_csv(OUTPUT, index=False, encoding="utf-8")
print(f"合并结果已保存：{OUTPUT}")
