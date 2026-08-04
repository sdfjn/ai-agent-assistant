"""
工具定义 —— 使用 LangChain 的 @tool 装饰器。

LangChain 规范：
  每个工具 = 一个 Python 函数 + @tool 装饰器
  @tool 自动完成：函数名→工具名、docstring→描述、类型注解→JSON Schema。

对比旧项目中手动写 TOOLS 字典的方式，
这里不需要手动维护 JSON Schema，代码更简洁可靠。
"""

import json
from datetime import datetime

import httpx
from langchain.tools import tool


# ─── 子函数：城市名 → 经纬度 ──────────────────────────────
# 工具内部使用的辅助函数，不需要 @tool


def _geocode_city(city_name: str) -> dict:
    """调用 Open-Meteo 免费地理编码 API，城市名转经纬度。"""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city_name, "count": 1, "language": "zh"}
    resp = httpx.get(url, params=params, timeout=10)
    data = resp.json()

    if data.get("results"):
        r = data["results"][0]
        return {
            "city": r.get("name", city_name),
            "country": r.get("country", ""),
            "latitude": r["latitude"],
            "longitude": r["longitude"],
        }
    return {"error": f"找不到城市: {city_name}"}


# ─── 天气码 → 中文描述 ────────────────────────────────────

_WEATHER_CODES: dict[int, str] = {
    0: "☀️ 晴天", 1: "🌤️ 大部晴朗", 2: "⛅ 多云",
    3: "☁️ 阴天", 45: "🌫️ 有雾", 48: "🌫️ 雾凇",
    51: "🌧️ 小雨", 53: "🌧️ 中雨", 55: "🌧️ 大雨",
    61: "🌧️ 小阵雨", 63: "🌧️ 中阵雨", 65: "🌧️ 大阵雨",
    71: "🌨️ 小雪", 73: "🌨️ 中雪", 75: "🌨️ 大雪",
    80: "🌧️ 小阵雨", 81: "🌧️ 中阵雨", 82: "🌧️ 大阵雨",
    95: "⛈️ 雷暴", 96: "⛈️ 雷暴+冰雹", 99: "⛈️ 强雷暴+冰雹",
}


# ═══════════════════════════════════════════════════════════
# 工具 1：天气查询
# ═══════════════════════════════════════════════════════════


@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的实时天气信息，包括温度、风速、天气状况。
    当用户询问天气、气温、会不会下雨等问题时使用此工具。

    Args:
        city: 城市名称，例如 '北京'、'上海'、'Tokyo'、'London'
    """
    # 步骤 1：城市名 → 经纬度
    geo = _geocode_city(city)
    if "error" in geo:
        return geo["error"]

    # 步骤 2：经纬度 → 天气
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": geo["latitude"],
        "longitude": geo["longitude"],
        "current_weather": True,
        "timezone": "auto",
    }
    resp = httpx.get(url, params=params, timeout=10)
    cw = resp.json().get("current_weather", {})

    weather_desc = _WEATHER_CODES.get(cw.get("weathercode", 0), "未知")

    return json.dumps({
        "城市": f"{geo['city']}, {geo['country']}",
        "温度": f"{cw.get('temperature', 'N/A')}°C",
        "风速": f"{cw.get('windspeed', 'N/A')} km/h",
        "天气": weather_desc,
        "观测时间": cw.get("time", "N/A"),
    }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════
# 工具 2：时间查询
# ═══════════════════════════════════════════════════════════


@tool
def get_time() -> str:
    """
    获取当前日期和时间。
    当用户询问现在几点、今天几号、星期几时使用此工具。
    """
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} {weekdays[now.weekday()]}"


# ─── 工具注册表 ──────────────────────────────────────────


# 所有内置工具的集合，供 Agent 工厂函数使用
BUILTIN_TOOLS: list = [get_weather, get_time]
