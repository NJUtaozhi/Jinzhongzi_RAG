from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import subprocess
import csv
import os
import uuid
import shutil

app = FastAPI(title="Vision Analysis Service")

OPENFACE_BIN = r"D:\学习资料\科研类项目\金种子早期科研\openface-venv\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FaceLandmarkImg.exe"

AU_NAMES = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r",
    "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r",
    "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"
]


def infer_emotion(au_dict):
    """
    基于 AU 组合规则推断基本情绪。
    - AU12_r(嘴角上扬) + AU06_r(脸颊抬起) → happiness
    - AU04_r(眉毛下压) + AU07_r(眼睑收紧) → anger
    - AU01_r(眉毛内侧抬起) + AU04_r + AU15_r(嘴角下撇) → sadness
    - AU01_r + AU02_r(眉毛外侧抬起) + AU05_r(上眼睑抬起) → surprise
    - AU09_r(鼻梁皱起) + AU10_r(上唇抬起) → disgust
    - AU01_r + AU04_r + AU20_r(嘴角拉伸) → fear
    其他情况 → neutral
    """
    au06 = au_dict.get("AU06_r", 0)
    au12 = au_dict.get("AU12_r", 0)
    au04 = au_dict.get("AU04_r", 0)
    au07 = au_dict.get("AU07_r", 0)
    au01 = au_dict.get("AU01_r", 0)
    au15 = au_dict.get("AU15_r", 0)
    au02 = au_dict.get("AU02_r", 0)
    au05 = au_dict.get("AU05_r", 0)
    au09 = au_dict.get("AU09_r", 0)
    au10 = au_dict.get("AU10_r", 0)
    au20 = au_dict.get("AU20_r", 0)

    if au12 > 1.0 and au06 > 0.5:
        return "happiness"
    elif au04 > 1.0 and au07 > 0.5:
        return "anger"
    elif au01 > 1.0 and au04 > 0.5 and au15 > 0.5:
        return "sadness"
    elif au01 > 0.5 and au02 > 0.5 and au05 > 0.5:
        return "surprise"
    elif au09 > 0.5 and au10 > 0.5:
        return "disgust"
    elif au01 > 0.5 and au04 > 0.5 and au20 > 0.5:
        return "fear"
    else:
        return "neutral"


def parse_openface_csv(csv_path):
    """
    解析 OpenFace 输出的 CSV，提取 AU 强度值。
    使用 skipinitialspace=True 处理列名前导空格；
    通过置信度 (confidence > 0.5) 过滤无效帧。
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            # OpenFace 2.2.0 输出包含 confidence 列，>0.5 表示检测有效
            try:
                conf = float(row.get("confidence", 0))
            except (ValueError, TypeError):
                conf = 0.0

            if conf > 0.5:
                au_dict = {}
                for au_name in AU_NAMES:
                    try:
                        au_dict[au_name] = float(row.get(au_name, 0))
                    except (ValueError, TypeError):
                        au_dict[au_name] = 0.0
                return au_dict
    return None


@app.post("/v1/vision/analyze-face")
async def analyze_face(image: UploadFile = File(...)):
    """
    接收人脸图片，调用 OpenFace 分析，返回 17 个 AU 强度值与情绪判断。

    请求：multipart/form-data，字段名 image
    响应：
    {
        "code": 200,
        "msg": "success",
        "data": {
            "face_detected": true,
            "dominant_emotion": "happiness",
            "au_analysis": { "AU01_r": 1.10, "AU02_r": 0.81, ... },
            "hint": "AU12_r (嘴角上扬) 数值较高，可能表示笑容。"
        }
    }
    """
    # 创建临时目录
    temp_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(os.getcwd(), f"temp_openface_{temp_id}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # 保存上传的图片
        image_path = os.path.join(temp_dir, image.filename or "input.jpg")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # 调用 OpenFace 命令行工具
        result = subprocess.run(
            [OPENFACE_BIN, "-f", image_path, "-out_dir", temp_dir],
            capture_output=True,
            text=True,
            timeout=60
        )

        # 查找输出的 CSV 文件
        csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
        if not csv_files:
            return JSONResponse(
                status_code=400,
                content={
                    "code": 40010,
                    "msg": "未检测到人脸，请提供包含清晰正面人脸的图片。",
                    "data": None
                }
            )

        csv_path = os.path.join(temp_dir, csv_files[0])
        au_dict = parse_openface_csv(csv_path)

        if au_dict is None:
            return JSONResponse(
                status_code=400,
                content={
                    "code": 40010,
                    "msg": "未检测到人脸，请提供包含清晰正面人脸的图片。",
                    "data": None
                }
            )

        # 推断情绪
        emotion = infer_emotion(au_dict)

        # 生成提示
        hints = []
        if au_dict.get("AU12_r", 0) > 1.5:
            hints.append("AU12_r (嘴角上扬) 数值较高，可能表示笑容。")
        if au_dict.get("AU04_r", 0) > 1.5:
            hints.append("AU04_r (眉毛下压) 数值较高，可能表示紧张或专注。")

        return {
            "code": 200,
            "msg": "success",
            "data": {
                "face_detected": True,
                "dominant_emotion": emotion,
                "au_analysis": au_dict,
                "hint": "；".join(hints) if hints else "面部表情分析完成。"
            }
        }

    except subprocess.TimeoutExpired:
        return JSONResponse(
            status_code=500,
            content={"code": 50000, "msg": "OpenFace 分析超时，请重试。", "data": None}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"code": 50000, "msg": f"服务内部错误：{str(e)}", "data": None}
        )
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/health")
def health_check():
    """健康检查接口"""
    return {"status": "ok", "service": "vision-analysis"}


