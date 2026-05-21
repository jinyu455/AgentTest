import uuid
from datetime import datetime

from flask import Flask, request, jsonify

from agents.chat_agent import ChatAgent
from app.agent_pipeline import AgentPipeline
from database.db import Database

app = Flask(__name__)
pipeline = AgentPipeline()
chat_agent = ChatAgent()

try:
    db = Database()
    db._connect()
    pipeline.db = db
    print("[app] 数据库连接成功")
except Exception as e:
    print(f"[app] 数据库连接失败{e}")


def build_input_data(body, source_default="api"):
    text = body.get("text")
    if not text or not isinstance(text, str) or not text.strip():
        raise ValueError("text 字段为必填")

    return {
        "id": body.get("id") or str(uuid.uuid4()),
        "user_id": body.get("user_id", "anonymous"),
        "text": text.strip(),
        "source": body.get("source") or source_default,
        "created_at": body.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_chat_input(analysis_result):
    return {
        "id": analysis_result.get("id", ""),
        "text": analysis_result.get("text", ""),
        "sample_type": analysis_result.get("sample_type", "direct"),
        "emotion": analysis_result.get("emotion", "中性"),
        "secondary_emotion": analysis_result.get("secondary_emotion"),
        "intensity": analysis_result.get("intensity", 50),
        "final_confidence": analysis_result.get("final_confidence", 0.5),
        "is_sarcasm": analysis_result.get("is_sarcasm", False),
        "is_mixed": analysis_result.get("is_mixed", False),
        "reason": analysis_result.get("reason", ""),
        "tokens": analysis_result.get("tokens", []),
        "emotion_words": analysis_result.get("emotion_words", []),
        "source": analysis_result.get("source", "chat"),
        "created_at": analysis_result.get("created_at", ""),
    }


@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return "", 204

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "请求体为空或非JSON格式"}), 400

    try:
        input_data = build_input_data(body)
        result = pipeline.run(input_data)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"分析失败: {str(e)}"}), 500


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 204

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "请求体为空或非JSON格式"}), 400

    try:
        input_data = build_input_data(body, source_default="chat")
        analysis_result = pipeline.run(input_data)
        chat_result = chat_agent.process(build_chat_input(analysis_result))
        return jsonify(chat_result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"聊天失败: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
