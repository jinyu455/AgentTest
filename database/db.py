import os
from datetime import datetime
from typing import Optional, Dict, Any
import json
import pymysql
from pymysql.cursors import DictCursor

class Database:

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 user: Optional[str] = None, password: Optional[str] = None,
                 database: Optional[str] = None):
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = port or int(os.getenv("DB_PORT", "3306"))
        self.user = user or os.getenv("DB_USER", "root")
        self.password = password or os.getenv("DB_PASSWORD", "123456")
        self.database = database or os.getenv("DB_NAME", "emotion_agent")
        self._conn: Optional[pymysql.Connection] = None

    def _connect(self) -> pymysql.Connection:
        if self._conn is None:
            self._conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset="utf8mb4",
                cursorclass=DictCursor,
            )
        return self._conn


    def _normalize_datetime(self, value: Any) -> Optional[str]:
        if value is None:
            return None

        text = str(value).strip()
        if not text:
            return None

        normalized = text.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return text.replace("T", " ").split(".", 1)[0]

    def insert_raw_text(self, input_data: Dict[str, Any]) -> None:
        conn = self._connect()
        with conn.cursor() as cursor:
            sql = """INSERT INTO raw_text (id, user_id, text, source, created_at)
                     VALUES (%s, %s, %s, %s, %s)"""
            cursor.execute(sql, (
                input_data["id"],
                input_data["user_id"],
                input_data["text"],
                input_data["source"],
                self._normalize_datetime(input_data.get("created_at")),
            ))
        conn.commit()

    def insert_emotion_result(self, result: Dict[str, Any]) -> None:
        conn = self._connect()
        with conn.cursor() as cursor:
            sql = """INSERT INTO emotion_result
                     (text_id, sample_type, emotion, secondary_emotion,
                      intensity, final_confidence, is_sarcasm, is_mixed,
                      reason, tokens, emotion_words, created_at)
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (
                result["id"],
                result["sample_type"],
                result["emotion"],
                result.get("secondary_emotion"),
                result["intensity"],
                result["final_confidence"],
                int(result["is_sarcasm"]),
                int(result["is_mixed"]),
                result.get("reason"),
                json.dumps(result.get("tokens", []), ensure_ascii=False),
                json.dumps(result.get("emotion_words", []), ensure_ascii=False),
                self._normalize_datetime(result.get("created_at")),
            ))
        conn.commit()
