import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    """数据库配置"""

    # 数据库连接
    HOST: str = os.getenv('DB_HOST', '36.103.203.75')
    PORT: int = int(os.getenv('DB_PORT', '1169'))
    USER: str = os.getenv('DB_USER', 'postgres')
    PASSWORD: str = os.getenv('DB_PASSWORD', 'ximukeji2026')
    DATABASE: str = os.getenv('DB_NAME', 'knowledge_rag_db')

    # 连接池配置
    POOL_SIZE: int = 10
    MAX_OVERFLOW: int = 0
    POOL_TIMEOUT: int = 30

    @property
    def url(self) -> str:
        return (
            f"postgresql+asyncpg://"
            f"{self.USER}:{self.PASSWORD}"
            f"@{self.HOST}:{self.PORT}/{self.DATABASE}"
        )

db_config = DatabaseConfig()