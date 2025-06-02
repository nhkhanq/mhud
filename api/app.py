from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib, pandas as pd, os, ast
from openai import OpenAI
from textwrap import dedent


# === 0. Ghi log ===
# --- thêm cuối cụm import ---
import uuid, datetime, pathlib

# --- khởi tạo thư mục log ---
LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def write_log(line: str):
    """Ghi một dòng (kèm timestamp) vào file log phiên hiện tại."""
    log_name = getattr(app, "log_name", "default_session.txt")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with (LOG_DIR / log_name).open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {line}\n")


# === 1. Cấu hình GPT-4o-mini ===
client = OpenAI(
    api_key="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"
)

# === 2. Load mô hình ML ===
model_dir = os.path.join(os.getcwd(), "models")
model = joblib.load(os.path.join(model_dir, "best_rf_model.pkl"))
label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.pkl"))
selected_features = joblib.load(os.path.join(model_dir, "selected_features.pkl"))

# --- 2.1 Chỉ giữ encoder cho 6 cột chữ ---
NEED_ENCODE = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "SCC",
    "SMOKE",
    "MTRANS",
]
label_encoders = {k: v for k, v in label_encoders.items() if k in NEED_ENCODE}

# --- 2.2 Các cột số (không encode) ---
NUMERIC_COLS = [
    "Age", "Height", "Weight",
    "FCVC", "NCP", "TUE", "FAF",
    "CH2O", "CAEC", "CALC",
]

# === 3. Flask App ===
app = Flask(__name__)
CORS(app)

# === 4. Trường thông tin cần cho ML ===
required_fields = [
    "Gender", "Age", "Height", "Weight", "family_history_with_overweight",
    "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF",
    "TUE", "CALC", "MTRANS"
]

# === 5. System Prompt cốt lõi (SYSTEM_CORE) – phiên bản đã được tinh chỉnh cho tự nhiên ===
SYSTEM_CORE = {
    "role": "system",
    "content": (
        "👩‍⚕️ Bạn là **Bác sĩ Dinh dưỡng AI** – hỗ trợ người dùng theo phong cách nhẹ nhàng, gần gũi, tự nhiên như một người bạn thân thiện.\n\n"
        "🎯 Mục tiêu:\n"
        "1. Trò chuyện để thu thập đủ **16 thông tin sức khỏe** dưới đây (dùng nội bộ để phân tích).\n"
        "2. Khi đủ, nhắc người dùng gõ **phân tích** để trả về kết quả ML + tư vấn chi tiết.\n\n"
        "🧠 Hướng dẫn phản hồi:\n"
        "• Bắt đầu bằng **một lời chào ấm áp, chân thành** – ví dụ:\n"
        "  “Chào bạn! 😊 Mình là bác sĩ dinh dưỡng AI, hôm nay đồng hành cùng bạn nè!”\n\n"
        "• **Tạo nhịp kết nối cảm xúc trước khi hỏi số liệu** – ví dụ:\n"
        "  “Bạn dạo này thấy sức khỏe ổn không?”, hoặc “Bạn đang có mục tiêu gì không – tăng cân, giảm cân hay chỉ muốn duy trì ha?”\n\n"
        "• **Chỉ khi người dùng phản hồi tích cực hoặc sẵn sàng** → mới xin phép bắt đầu hỏi: \n"
        "  “Nếu bạn thấy thoải mái, mình hỏi bạn vài điều nhỏ để hỗ trợ tốt hơn nha.”\n\n"
        "• Khi hỏi từng mục:\n"
        "  – Dẫn dắt ngắn gọn, tự nhiên (không gấp gáp). Ví dụ:\n"
        "    “Mình có thể biết tuổi của bạn trước được không nè?” hoặc “Bạn cao tầm bao nhiêu cm ha?”\n"
        "  – Tuyệt đối không hỏi 2–3 mục liền nhau.\n"
        "  – Giữa mỗi lần nên có nhịp trò chuyện: ví dụ “Ok, để mình ghi lại nhé.” hoặc “Cảm ơn bạn đã chia sẻ!”\n\n"
        "• Nếu người dùng trả lời mơ hồ hoặc sai phạm vi ➜ đoán nhẹ và xác nhận lại:\n"
        "  _“Bạn định là khoảng 2 buổi tập/tuần – đúng không?”_\n\n"
        "• Khi nhận thông tin rõ ràng ➜ xác nhận nhẹ nhàng:\n"
        "  _“Mình đã ghi CH2O = 2 lít/ngày – đúng không nè?”_\n\n"
        "• Nếu còn thiếu ➜ chọn tối đa 1–2 mục đầu danh sách thiếu để hỏi tiếp (ưu tiên Height, Weight nếu trống).\n\n"
        "• Khi đủ 16 mục ➜ nhắc người dùng gõ **phân tích** để xem kết quả và nhận lời khuyên cá nhân hóa.\n\n"
        "⚠️ Không bao giờ dùng cụm từ như 'trường dữ liệu', 'form', 'mẫu thông tin'…\n"
        "⚠️ Không in ra danh sách các biến số – chỉ dùng nội bộ.\n"
        "⚠️ Giọng nói phải **gợi cảm giác đang được quan tâm**, chứ không phải đang làm khảo sát.\n\n"
        "📋 Dữ liệu cần lấy (16 mục nội bộ):\n"
        "- Age: Tuổi (năm; >0)\n"
        "- Height: Chiều cao (cm; >0)\n"
        "- Weight: Cân nặng (kg; >0)\n"
        "- FCVC: Ăn rau (1=Không, 2=Thỉnh thoảng, 3=Luôn)\n"
        "- Gender: Male / Female\n"
        "- NCP: Bữa chính/ngày (≥1)\n"
        "- TUE: Thời gian thiết bị/ngày (≥0 giờ)\n"
        "- FAF: Ngày tập thể dục/tuần (0–7)\n"
        "- CH2O: Lượng nước uống/ngày (lít; >0)\n"
        "- CAEC: Ăn vặt (0–3)\n"
        "- CALC: Rượu bia (0–3)\n"
        "- family_history_with_overweight: Yes / No\n"
        "- MTRANS: Phương tiện di chuyển (Walking, Bike, Motorbike, Public_Transportation, Car)\n"
        "- FAVC: Thích đồ calo cao (Yes / No)\n"
        "- SCC: Có ghi calo hàng ngày không? (Yes / No)\n"
        "- SMOKE: Hút thuốc? (Yes / No)\n"
    )
}

session_messages = [SYSTEM_CORE]

# === 6. Biến trạng thái chờ xác nhận ===
# Khi người dùng gõ "phân tích" mà đủ 16 trường, ta gửi summary rồi set this flag = True.
# Khi next request là "xác nhận", mới chạy predict.
awaiting_confirmation = False

# === 6b. Thông tin cố định cho từng nhãn ===
OBESITY_INFO = {
    "Insufficient_Weight": {
        "vi_name": "Thiếu cân",
        "bmi": "< 18.5",
        "risks": [
            "Suy giảm miễn dịch, dễ nhiễm bệnh",
            "Thiếu vi chất (sắt, kẽm, vitamin D)",
            "Giảm mật độ xương, nguy cơ loãng xương"
        ],
        "goal": "Tăng 0.3–0.5 kg/tuần bằng dinh dưỡng giàu đạm & tập kháng lực nhẹ",
    },
    "Normal_Weight": {
        "vi_name": "Cân nặng bình thường",
        "bmi": "18.5 – 24.9",
        "risks": ["Tiếp tục duy trì lối sống lành mạnh"],
        "goal": "Giữ BMI & vòng eo tối ưu; ≥ 150 phút vận động/tuần",
    },
    "Overweight_Level_I": {
        "vi_name": "Thừa cân mức I",
        "bmi": "25 – 27.4",
        "risks": [
            "Mỡ máu, huyết áp tăng nhẹ",
            "Ngưng thở khi ngủ (nhẹ)"
        ],
        "goal": "Giảm 5 % cân nặng (~0.5 kg/tuần trong 3 tháng)",
    },
    "Overweight_Level_II": {
        "vi_name": "Thừa cân mức II",
        "bmi": "27.5 – 29.9",
        "risks": ["Gan nhiễm mỡ sớm, tiền đái tháo đường"],
        "goal": "Giảm 7 % cân nặng; vòng eo < 90 cm (nam) / 80 cm (nữ)",
    },
    "Obesity_Type_I": {
        "vi_name": "Béo phì loại I",
        "bmi": "30 – 34.9",
        "risks": [
            "Nguy cơ bệnh tim mạch tăng gấp 2",
            "Đau khớp gối, thắt lưng"
        ],
        "goal": "Giảm 10 % cân nặng trong 6 tháng; theo dõi huyết áp 3 tháng/lần",
    },
    "Obesity_Type_II": {
        "vi_name": "Béo phì loại II",
        "bmi": "35 – 39.9",
        "risks": [
            "Tiểu đường type 2, hội chứng chuyển hóa",
            "Ngưng thở khi ngủ mức vừa"
        ],
        "goal": "Giảm 10–15 % cân nặng; cân nhắc thuốc hỗ trợ",
    },
    "Obesity_Type_III": {
        "vi_name": "Béo phì nghiêm trọng",
        "bmi": "≥ 40",
        "risks": [
            "Nguy cơ tử vong do tim mạch tăng cao",
            "Thoái hóa khớp nặng, trầm cảm"
        ],
        "goal": "Giảm ≥ 15 % cân nặng; xem xét phẫu thuật giảm cân",
    },
}


# === 7. Hàm dự đoán (predict_obesity) ===
def predict_obesity(input_data: dict) -> str:
    """
    Nhận dict 16 trường, ép kiểu chuẩn,
    encode 6 cột chữ, sau đó dự đoán.
    """
    df = pd.DataFrame([input_data])

    # 6 cột chữ → dùng LabelEncoder (xử lý giá trị lạ an toàn)
    for col in NEED_ENCODE:
        if col in df.columns:
            enc = label_encoders[col]
            # Nếu giá trị chưa nằm trong enc.classes_, gán về class_0 để tránh ValueError
            df[col] = df[col].apply(
                lambda x: x if x in enc.classes_ else enc.classes_[0]
            )
            df[col] = enc.transform(df[col])

    # Các cột số → ép kiểu int/float
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Giữ đúng thứ tự selected_features
    df = df[selected_features]

    return model.predict(df)[0]

# === 8. Sinh gói tư vấn chi tiết ===
def generate_advice(label: str) -> str:
    """
    Tạo lời khuyên gồm: ý nghĩa mức độ, nguy cơ, mục tiêu,
    3 thói quen, thực đơn 1 ngày, lịch tập 7 ngày, cách theo dõi.
    """
    info = OBESITY_INFO.get(label, {})
    vi_name  = info.get("vi_name", label)
    bmi_txt  = info.get("bmi", "—")
    risks_md = "\n".join(f"• {r}" for r in info.get("risks", []))
    goal_txt = info.get("goal", "Duy trì lối sống lành mạnh")

    user_prompt = dedent(f"""
        Bạn đang ở nhóm **{vi_name}** (BMI {bmi_txt}).
        Nguy cơ sức khỏe chính:
        {risks_md}

        🎯 **Mục tiêu**: {goal_txt}

        Hãy cho tôi:
        1. Giải thích ngắn gọn vì sao tôi ở nhóm này.
        2. 3 thói quen cần áp dụng ngay (đánh số ①②③).
        3. Thực đơn gợi ý 1 ngày (3 bữa chính + 2 phụ) kèm kcal.
        4. Lịch tập 7 ngày mức độ vừa (ghi rõ phút & cường độ).
        5. Cách tự theo dõi tiến trình & mốc tái khám.

        Trình bày thân thiện, xen emoji phù hợp, súc tích, dễ thực thi.
    """)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Bạn là chuyên gia dinh dưỡng & huấn luyện sức khỏe, ưu tiên lời khuyên thực tiễn ngắn gọn."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


# === 9. Trích dict 16 trường từ lịch sử ===
def extract_model_input_from_history() -> dict:
    prompt = {
        "role": "user",
        "content": (
            "Dựa trên toàn bộ cuộc hội thoại dưới đây giữa bạn và người dùng, "
            "hãy trả về một dict Python chỉ gồm những khóa sau:\n\n"
            f"{', '.join(required_fields)}\n\n"
            "Giá trị phải đúng kiểu:\n"
            "- Các trường số → int hoặc float\n"
            "- Gender → 'Male' hoặc 'Female'\n"
            "- Yes/No → 'Yes' hoặc 'No'\n"
            "- MTRANS → đúng một trong: Walking, Bike, Motorbike, Public_Transportation, Car\n\n"
            "⚠️ Chỉ trả về `dict Python`, không chú thích, không lời giải thích. Ví dụ:\n"
            "{'Gender': 'Male', 'Age': 22, 'Height': 170, ...}\n\n"
            "⚠️ Nếu thiếu trường nào, hãy **bỏ luôn khóa đó** trong dict.\n"
            "Bắt đầu trả kết quả ở dòng sau:"
        )
    }

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=session_messages + [prompt],
        temperature=0,
    )

    try:
        return ast.literal_eval(resp.choices[0].message.content.strip())
    except Exception as e:
        print("⚠️ Lỗi parse:", e)
        return {}


# === 10. Prompt ẩn động theo từng lượt (SYSTEM-FLOW) ===
def build_flow_prompt(current: dict, missing: list[str]) -> dict:
    row = [str(current.get(col, "")) for col in required_fields]
    table = "\t".join(required_fields) + "\n" + "\t".join(row)

    definitions = (
        "📋 Định nghĩa (nội bộ, không hiện ra ngoài):\n"
        "- Age: số nguyên > 0\n"
        "- FCVC: 1–3 (mức độ ăn rau)\n"
        "- Gender: Male/Female\n"
        "- NCP: ≥1\n"
        "- TUE: ≥0\n"
        "- FAF: 0–7\n"
        "- CH2O: >0\n"
        "- CAEC, CALC: 0–3\n"
        "- family_history_with_overweight, FAVC, SCC, SMOKE: Yes/No\n"
        "- MTRANS: Walking/Bike/Motorbike/Public_Transportation/Car"
    )

    instructions = (
        "🎯 Hướng dẫn xử lý:\n"
        "• Nếu `MISSING` rỗng → nhắc người dùng gõ **phân tích**, không hỏi gì thêm.\n"
        "• Nếu còn thiếu → hỏi tối đa 1–2 trường đầu (ưu tiên Height, Weight nếu chưa có).\n"
        "• Khi nhận giá trị, xác nhận: _“Mình đã ghi FAF = 3 ngày/tuần – đúng không?”_\n"
        "• Nếu giá trị mơ hồ → hỏi lại nhẹ nhàng kèm ví dụ.\n"
        "• Giữ ngôn ngữ tự nhiên, như một bác sĩ tư vấn nhẹ nhàng."
    )

    # 👉  Trả về với role = 'assistant' để KHÔNG đè system-prompt chính
    return {
        "role": "assistant",
        "content": (
            f"### SYSTEM FLOW (ẨN) ###\n"
            f"{definitions}\n\n"
            f"DATA:\n{table}\n\n"
            f"MISSING: {missing}\n\n"
            f"{instructions}"
        )
    }


# === 11. Hàm chat chính ===
def chat_with_gpt(user_input: str) -> str:
    # Lưu lời user & log
    session_messages.append({"role": "user", "content": user_input})
    write_log(f"USER: {user_input}")

    # Phân tích trạng thái hiện tại
    current = extract_model_input_from_history()
    missing = [f for f in required_fields if f not in current or current[f] in ["", None]]
    flow_prompt = build_flow_prompt(current, missing)  # role = assistant

    messages = [SYSTEM_CORE] + session_messages[1:] + [flow_prompt]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.5,
    )

    reply = resp.choices[0].message.content.strip()
    session_messages.append({"role": "assistant", "content": reply})
    write_log(f"ASSISTANT: {reply}")        # 📝 log assistant

    return reply


# === 12. Flask routes ===
@app.route("/chat")
@app.route("/")
def chat_ui():
    global session_messages, awaiting_confirmation

    # 🔄 RESET phiên mỗi lần tải trang
    session_messages = [SYSTEM_CORE]
    awaiting_confirmation = False

    # 📂 tạo file log mới cho phiên này
    app.log_name = f"chat_{uuid.uuid4().hex[:8]}.txt"
    write_log("=== NEW SESSION ===")

    return render_template("chatgpt_ui.html")


# === 12. Flask routes ===
@app.route("/ask", methods=["POST"])
def ask():
    global session_messages, awaiting_confirmation

    user_input = request.json.get("message", "").strip()
    write_log(f"USER: {user_input}")        # 📝 log phát ngôn người dùng

    # 1️⃣  ĐANG CHỜ XÁC NHẬN
    if awaiting_confirmation and user_input.lower() in ["xác nhận", "xac nhan", "confirm"]:
        user_data = extract_model_input_from_history()
        write_log(f"MODEL_INPUT: {user_data}")   # log dict gửi vào RF

        label = predict_obesity(user_data)
        write_log(f"PREDICTED_LABEL: {label}")   # log nhãn dự đoán

        advice = generate_advice(label)
        final_msg = f"📊 **Kết quả**\n\nBạn thuộc nhóm **{label}**.\n\n{advice}"
        session_messages.append({"role": "assistant", "content": final_msg})
        write_log(f"ASSISTANT: {final_msg}")     # log phản hồi bot

        awaiting_confirmation = False            # reset cờ
        return jsonify({"reply": final_msg})

    # 2️⃣  NGƯỜI DÙNG GÕ “PHÂN TÍCH”
    if user_input.lower() in ["phân tích", "phan tich", "pt", "tiếp tục"]:
        user_data = extract_model_input_from_history()
        missing = [f for f in required_fields if f not in user_data or user_data[f] in ["", None]]

        # 2.1  Thiếu thông tin
        if missing:
            reply = f"⚠️ Mình còn thiếu: {', '.join(missing)}. Bạn bổ sung nhé!"
            session_messages.append({"role": "assistant", "content": reply})
            write_log(f"ASSISTANT: {reply}")     # log phản hồi bot
            return jsonify({"reply": reply})

        # 2.2  Đã đủ 16 trường – tạo tóm tắt
        nice_map = {
            "Gender": "Giới tính", "Age": "Tuổi", "Height": "Chiều cao", "Weight": "Cân nặng",
            "family_history_with_overweight": "Tiền sử thừa cân gia đình",
            "FAVC": "Thích đồ calo cao", "FCVC": "Mức độ ăn rau (1-3)",
            "NCP": "Bữa chính/ngày", "CAEC": "Mức độ ăn vặt (0-3)",
            "SMOKE": "Hút thuốc", "CH2O": "Lượng nước (lít/ngày)",
            "SCC": "Ghi chép calo", "FAF": "Ngày tập/tuần",
            "TUE": "Giờ dùng thiết bị/ngày", "CALC": "Mức độ uống rượu/bia (0-3)",
            "MTRANS": "Phương tiện di chuyển"
        }
        summary_lines = []
        for key in required_fields:
            val = user_data.get(key)
            if isinstance(val, str) and val.lower() in ["yes", "no"]:
                val = "Có" if val.lower() == "yes" else "Không"
            summary_lines.append(f"- {nice_map.get(key, key)}: {val}")

        summary = (
            "📝 **Tóm tắt thông tin bạn đã cung cấp:**\n" +
            "\n".join(summary_lines) +
            "\n\nNếu bạn muốn điều chỉnh bất kỳ thông tin nào, vui lòng nhập lại giá trị tương ứng.\n"
            "Nếu mọi thứ đã đúng, bạn gõ **xác nhận** để mình tiến hành phân tích nhé!"
        )

        awaiting_confirmation = True
        session_messages.append({"role": "assistant", "content": summary})
        write_log(f"ASSISTANT: {summary}")       # log phản hồi bot
        return jsonify({"reply": summary})

    # 3️⃣  TRƯỜNG HỢP BÌNH THƯỜNG – tiếp tục hội thoại
    reply = chat_with_gpt(user_input)            # chat_with_gpt đã tự log ASSISTANT
    return jsonify({"reply": reply})


# === 13. Chạy app ===
if __name__ == "__main__":
    app.run(debug=True)


from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)

def handler(request, response):
    return app(request.environ, response)
