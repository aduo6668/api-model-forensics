from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
MODELS_DIR = APP_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
RUNS_DIR = OUTPUTS_DIR / "runs"
LOGS_DIR = ROOT_DIR / "logs"
ENV_FILE = ROOT_DIR / ".env"
SCHEMA_VERSION = "0.1"
TOOL_VERSION = "0.1.0"
PRIVACY_MODE = "local-only"

PRIVACY_README_TEXT = (
    "本工具默认仅在用户本机运行。检测输入、输出结果、日志和 PDF 报告默认只保存在用户本地，"
    "不会主动上传到任何云端服务。API Key 由用户自行提供和控制，报告文件也由用户自行决定是否分享。"
    "本工具提供的是概率性模型取证结论，用于辅助识别降级、混模或挂羊头卖狗肉风险，不承诺 100% 还原真实底模。"
)

PRIVACY_UI_TEXT = (
    "默认本地处理，数据不出设备。你的 API Key、检测输入、输出结果和报告文件都由你自己掌控，"
    "本工具不会主动把任何检测数据上传到云端。"
)

REPORT_FOOTER_TEXT = (
    "本报告由本地工具生成，检测数据默认仅保存在用户设备上，不上传云端。"
    "API Key、报告与相关文件均由用户自行控制。"
    "报告提供的是概率性证据分析，不保证 100% 识别真实底模。"
)

BUDGET_PROFILES = {
    "fast": {
        "label": "Fast / 低成本",
        "estimated_total_tokens": 12000,
        "max_total_tokens": 15000,
    },
    "standard": {
        "label": "Standard / 推荐",
        "estimated_total_tokens": 28000,
        "max_total_tokens": 40000,
    },
    "deep": {
        "label": "Deep / 更稳但更贵",
        "estimated_total_tokens": 52000,
        "max_total_tokens": 80000,
    },
}

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyh.ttf"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]


def ensure_runtime_dirs() -> None:
    for path in (OUTPUTS_DIR, REPORTS_DIR, RUNS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_font_path() -> Path | None:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_local_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_FILE.exists():
        return values
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def default_runtime_settings() -> dict[str, str]:
    local_env = load_local_env()
    return {
        "base_url": local_env.get("API_FORENSICS_BASE_URL", "mock://openai"),
        "api_key": local_env.get("API_FORENSICS_API_KEY", "demo-key"),
        "model": local_env.get("API_FORENSICS_MODEL", "gpt-4o"),
        "provider_hint": local_env.get("API_FORENSICS_PROVIDER_HINT", ""),
        "mode": local_env.get("API_FORENSICS_MODE", "standard"),
    }
