def get_aqi_category(aqi):
    """
    Return AQI category and display color.
    """

    aqi = float(aqi)

    if aqi <= 50:
        return {
            "category": "Good",
            "color": "#00E400",
            "emoji": "🟢"
        }

    elif aqi <= 100:
        return {
            "category": "Moderate",
            "color": "#FFFF00",
            "emoji": "🟡"
        }

    elif aqi <= 150:
        return {
            "category": "Unhealthy for Sensitive Groups",
            "color": "#FF7E00",
            "emoji": "🟠"
        }

    elif aqi <= 200:
        return {
            "category": "Unhealthy",
            "color": "#FF0000",
            "emoji": "🔴"
        }

    elif aqi <= 300:
        return {
            "category": "Very Unhealthy",
            "color": "#8F3F97",
            "emoji": "🟣"
        }

    else:
        return {
            "category": "Hazardous",
            "color": "#7E0023",
            "emoji": "🟤"
        }


def get_aqi_status_text(aqi):
    """
    Return formatted AQI status.
    """

    info = get_aqi_category(aqi)

    return f"{info['emoji']} {info['category']}"