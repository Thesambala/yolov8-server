"""astraea-vision config — semua dari environment, ada default prototype (PRD §14/17/49)."""
import os


def _getenv(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _getfloat(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _getint(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# --- model ---
MODEL_PATH = _getenv("DETECT_MODEL", "/home/ubuntu/yolov8s_indotraffic_best.pt")
CONFIDENCE = _getfloat("DETECT_CONF", 0.30)
IOU = _getfloat("DETECT_IOU", 0.60)
INFER_IMGSZ = _getint("DETECT_IMGSZ", 960)
MIN_SRC_WIDTH = _getint("DETECT_MIN_SRC_WIDTH", 960)

# --- ingest ---
WS_HOST = _getenv("VISION_WS_HOST", "0.0.0.0")
WS_PORT = _getint("VISION_WS_PORT", 8082)
WS_PATH = "/v1/camera/ingest"
MAX_FRAME_BYTES = 16 * 1024 * 1024
AUTH_CACHE_S = _getint("CAMERA_AUTH_CACHE_S", 60)

# --- http api ---
HTTP_HOST = _getenv("VISION_HTTP_HOST", "0.0.0.0")
HTTP_PORT = _getint("VISION_HTTP_PORT", 8081)
VIEWER_TOKEN = _getenv("VISION_VIEWER_TOKEN", "")

# --- freshness / scheduling (PRD §16/49) ---
VISION_FRESH_S = _getfloat("VISION_FRESH_S", 15.0)
METRIC_HZ = _getfloat("VISION_METRIC_HZ", 1.0)
FUZZY_MIN_INTERVAL_S = _getfloat("FUZZY_MIN_INTERVAL_S", 1.0)

# --- mqtt ---
MQTT_HOST = _getenv("MQTT_HOST", "13.238.154.250")
MQTT_PORT = _getint("MQTT_PORT", 1883)
MQTT_USER = _getenv("MQTT_USER", "jti")
MQTT_PASS = _getenv("MQTT_PASS", "")
MQTT_KEEPALIVE = _getint("MQTT_KEEPALIVE", 30)

# --- aws ---
AWS_REGION = _getenv("AWS_REGION", "ap-southeast-2")
CAMERAS_TABLE = _getenv("CAMERAS_TABLE", "Cameras")

# --- fuzzy bounds default (per-intersection config menimpa) ---
MIN_GREEN_S = _getfloat("MIN_GREEN_SEC", 10.0)
MAX_GREEN_S = _getfloat("MAX_GREEN_SEC", 60.0)
YELLOW_S = _getfloat("YELLOW_SEC", 3.0)
ALL_RED_S = _getfloat("ALL_RED_SEC", 2.0)
RECOMMENDATION_VALID_MS = _getint("RECOMMENDATION_VALID_MS", 10000)
QUEUE_CAPACITY_REF = _getfloat("QUEUE_CAPACITY_REF", 10.0)

SCHEMA_VERSION = 1
