#!/usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

from config import db_config


async def test_db_connection():
    """使用 config.py 中的配置测试 PostgreSQL 连接。"""
    try:
        import asyncpg
    except ImportError:
        print("未安装 asyncpg，请先安装：pip install asyncpg")
        return

    conn = None
    try:
        conn = await asyncpg.connect(
            host=db_config.HOST,
            port=db_config.PORT,
            user=db_config.USER,
            password=db_config.PASSWORD,
            database=db_config.DATABASE,
        )
        row = await conn.fetchrow(
            "SELECT current_database() AS db, current_user AS user_name, now() AS ts"
        )

        print("数据库连接成功")
        print(f"Host: {db_config.HOST}:{db_config.PORT}")
        print(f"Database: {row['db']}")
        print(f"User: {row['user_name']}")
        print(f"Server Time: {row['ts']}")
    except Exception as e:
        print("数据库连接失败")
        print(f"错误信息: {e}")
    finally:
        if conn is not None:
            await conn.close()


def main():
    asyncio.run(test_db_connection())


if __name__ == "__main__":
    main()
