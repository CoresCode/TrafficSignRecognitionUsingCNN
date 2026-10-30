"""
============================================================
GTSRB - German Traffic Sign Dataset Class Metadata Mapping
============================================================
Provides mapping for all 43 classes of traffic signs including:
- Name (English & German)
- Category (Speed Limit, Prohibitory, Mandatory, Warning, Priority)
- Traffic Rule / Description
- Category Color & Icon
"""

GTSRB_CLASSES = {
    0: {
        "name": "Speed limit (20km/h)",
        "german_name": "Zulässige Höchstgeschwindigkeit (20 km/h)",
        "category": "Speed Limit",
        "icon": "🛑",
        "description": "Maximum allowed speed is 20 kilometers per hour."
    },
    1: {
        "name": "Speed limit (30km/h)",
        "german_name": "Zulässige Höchstgeschwindigkeit (30 km/h)",
        "category": "Speed Limit",
        "icon": "⚡",
        "description": "Maximum allowed speed is 30 kilometers per hour. Commonly found in residential areas."
    },
    2: {
        "name": "Speed limit (50km/h)",
        "german_name": "Zulässige Höchstgeschwindigkeit (50 km/h)",
        "category": "Speed Limit",
        "icon": "🚗",
        "description": "Standard urban area speed limit of 50 km/h."
    },
    3: {
        "name": "Speed limit (60km/h)",
        "german_name": "Zulässige Höchstgeschwindigkeit (60 km/h)",
        "category": "Speed Limit",
        "icon": "🚙",
        "description": "Maximum speed limit 60 km/h."
    },
    4: {
        "name": "Speed limit (70km/h)",
        "german_name": "Zulässige Höchstgeschwindigkeit (70 km/h)",
        "category": "Speed Limit",
        "icon": "🏎️",
        "description": "Maximum speed limit 70 km/h."
    },
    5: {
        "name": "Speed limit (80km/h)",
        "german_name": "Zulässige Höchstgeschwindigkeit (80 km/h)",
        "category": "Speed Limit",
        "icon": "🏎️",
        "description": "Maximum speed limit 80 km/h. Standard speed on multi-lane urban expressways."
    },
    6: {
        "name": "End of speed limit (80km/h)",
        "german_name": "Ende der zulässigen Höchstgeschwindigkeit (80 km/h)",
        "category": "De-restriction",
        "icon": "🏁",
        "description": "80 km/h speed restriction ends."
    },
    7: {
        "name": "Speed limit (100km/h)",
        "german_name": "Zulässige Höchstgeschwindigkeit (100 km/h)",
        "category": "Speed Limit",
        "icon": "🚀",
        "description": "Maximum speed limit 100 km/h. Standard speed limit on rural highways in Germany."
    },
    8: {
        "name": "Speed limit (120km/h)",
        "german_name": "Zulässige Höchstgeschwindigkeit (120 km/h)",
        "category": "Speed Limit",
        "icon": "🚀",
        "description": "Maximum speed limit 120 km/h. Standard Autobahn speed limit zone."
    },
    9: {
        "name": "No passing",
        "german_name": "Überholverbot für Kraftfahrzeuge aller Art",
        "category": "Prohibitory",
        "icon": "🚫",
        "description": "Overtaking/passing other motor vehicles is strictly prohibited."
    },
    10: {
        "name": "No passing for vehicles over 3.5 metric tons",
        "german_name": "Überholverbot für Kraftfahrzeuge mit einer zulässigen Gesamtmasse über 3,5 t",
        "category": "Prohibitory",
        "icon": "🚛",
        "description": "Heavy goods vehicles weighing over 3.5t are forbidden from overtaking."
    },
    11: {
        "name": "Right-of-way at the next intersection",
        "german_name": "Vorfahrt an der nächsten Kreuzung oder Einmündung",
        "category": "Priority",
        "icon": "⚠️",
        "description": "Drivers on this road have priority right-of-way at the upcoming intersection."
    },
    12: {
        "name": "Priority road",
        "german_name": "Vorfahrtstraße",
        "category": "Priority",
        "icon": "🔷",
        "description": "Grants priority right-of-way to drivers along this main road at all oncoming intersections."
    },
    13: {
        "name": "Yield",
        "german_name": "Vorfahrt gewähren!",
        "category": "Priority",
        "icon": "🔻",
        "description": "Drivers must yield right-of-way to traffic on the main intersecting road."
    },
    14: {
        "name": "Stop",
        "german_name": "Halt! Vorfahrt gewähren!",
        "category": "Priority",
        "icon": "🛑",
        "description": "Drivers must come to a complete stop and yield to all cross traffic before proceeding."
    },
    15: {
        "name": "No vehicles",
        "german_name": "Verbot für Fahrzeuge aller Art",
        "category": "Prohibitory",
        "icon": "⛔",
        "description": "All vehicles are prohibited from entering in both directions."
    },
    16: {
        "name": "Vehicles over 3.5 metric tons prohibited",
        "german_name": "Verbot für Kraftfahrzeuge mit einer zulässigen Gesamtmasse über 3,5 t",
        "category": "Prohibitory",
        "icon": "🚛",
        "description": "Trucks and heavy commercial vehicles over 3.5t are forbidden."
    },
    17: {
        "name": "No entry",
        "german_name": "Verbot der Einfahrt",
        "category": "Prohibitory",
        "icon": "⛔",
        "description": "One-way street or restricted access. Do not enter from this direction."
    },
    18: {
        "name": "General caution",
        "german_name": "Gefahrstelle",
        "category": "Warning",
        "icon": "⚠️",
        "description": "General hazard alert ahead. Drive with caution."
    },
    19: {
        "name": "Dangerous curve to the left",
        "german_name": "Kurve (links)",
        "category": "Warning",
        "icon": "↩️",
        "description": "Warning of a sharp curve to the left ahead. Reduce speed."
    },
    20: {
        "name": "Dangerous curve to the right",
        "german_name": "Kurve (rechts)",
        "category": "Warning",
        "icon": "↪️",
        "description": "Warning of a sharp curve to the right ahead. Reduce speed."
    },
    21: {
        "name": "Double curve",
        "german_name": "Doppelkurve",
        "category": "Warning",
        "icon": "🔀",
        "description": "Warning of consecutive curves ahead."
    },
    22: {
        "name": "Bumpy road",
        "german_name": "Unebene Fahrbahn",
        "category": "Warning",
        "icon": "〰️",
        "description": "Uneven or damaged road surface ahead."
    },
    23: {
        "name": "Slippery road",
        "german_name": "Schleuder- oder Rutschgefahr",
        "category": "Warning",
        "icon": "🌧️",
        "description": "Road condition prone to skidding or slipping when wet or icy."
    },
    24: {
        "name": "Road narrows on the right",
        "german_name": "Einseitig (rechts) verengte Fahrbahn",
        "category": "Warning",
        "icon": "🔽",
        "description": "Carriageway narrows on the right side."
    },
    25: {
        "name": "Road work",
        "german_name": "Baustelle",
        "category": "Warning",
        "icon": "🚧",
        "description": "Construction or maintenance work underway ahead. Expect reduced speeds."
    },
    26: {
        "name": "Traffic signals",
        "german_name": "Lichtzeichenanlage",
        "category": "Warning",
        "icon": "🚦",
        "description": "Traffic lights control traffic ahead."
    },
    27: {
        "name": "Pedestrians",
        "german_name": "Fußgänger",
        "category": "Warning",
        "icon": "🚶",
        "description": "Warning of pedestrians in or near the roadway."
    },
    28: {
        "name": "Children crossing",
        "german_name": "Kinder",
        "category": "Warning",
        "icon": "🚸",
        "description": "Children crossing ahead near schools or playgrounds. Drive slowly."
    },
    29: {
        "name": "Bicycles crossing",
        "german_name": "Radverkehr",
        "category": "Warning",
        "icon": "🚲",
        "description": "Cyclists crossing or entering the road ahead."
    },
    30: {
        "name": "Beware of ice/snow",
        "german_name": "Schnee- oder Eisglätte",
        "category": "Warning",
        "icon": "❄️",
        "description": "Hazardous road conditions due to ice or snow."
    },
    31: {
        "name": "Wild animals crossing",
        "german_name": "Wildwechsel",
        "category": "Warning",
        "icon": "🦌",
        "description": "Frequent wildlife crossings ahead (deer, boars, etc.)."
    },
    32: {
        "name": "End of all speed and passing limits",
        "german_name": "Ende sämtlicher Streckenverbote",
        "category": "De-restriction",
        "icon": "🏁",
        "description": "End of all temporary speed limits and overtaking prohibitions."
    },
    33: {
        "name": "Turn right ahead",
        "german_name": "Fahrtrichtung hier rechts",
        "category": "Mandatory",
        "icon": "➡️",
        "description": "Drivers must turn right at the intersection ahead."
    },
    34: {
        "name": "Turn left ahead",
        "german_name": "Fahrtrichtung hier links",
        "category": "Mandatory",
        "icon": "⬅️",
        "description": "Drivers must turn left at the intersection ahead."
    },
    35: {
        "name": "Ahead only",
        "german_name": "Vorgeschriebene Fahrtrichtung geradeaus",
        "category": "Mandatory",
        "icon": "⬆️",
        "description": "Traffic must proceed straight ahead only."
    },
    36: {
        "name": "Go straight or right",
        "german_name": "Vorgeschriebene Fahrtrichtung geradeaus oder rechts",
        "category": "Mandatory",
        "icon": "↗️",
        "description": "Drivers may only proceed straight or turn right."
    },
    37: {
        "name": "Go straight or left",
        "german_name": "Vorgeschriebene Fahrtrichtung geradeaus oder links",
        "category": "Mandatory",
        "icon": "↖️",
        "description": "Drivers may only proceed straight or turn left."
    },
    38: {
        "name": "Keep right",
        "german_name": "Vorgeschriebene Vorbeifahrt rechts",
        "category": "Mandatory",
        "icon": "↘️",
        "description": "Pass the obstacle or traffic island on the right side."
    },
    39: {
        "name": "Keep left",
        "german_name": "Vorgeschriebene Vorbeifahrt links",
        "category": "Mandatory",
        "icon": "↙️",
        "description": "Pass the obstacle or traffic island on the left side."
    },
    40: {
        "name": "Roundabout mandatory",
        "german_name": "Kreisverkehr",
        "category": "Mandatory",
        "icon": "🔄",
        "description": "Mandatory roundabout ahead. Yield to traffic already inside the roundabout."
    },
    41: {
        "name": "End of no passing",
        "german_name": "Ende des Überholverbotes für Kraftfahrzeuge aller Art",
        "category": "De-restriction",
        "icon": "🏁",
        "description": "Prohibition of overtaking ends."
    },
    42: {
        "name": "End of no passing by vehicles over 3.5 metric tons",
        "german_name": "Ende des Überholverbotes für Kraftfahrzeuge mit einer Gesamtmasse über 3,5 t",
        "category": "De-restriction",
        "icon": "🏁",
        "description": "Overtaking prohibition for heavy goods vehicles ends."
    }
}


def get_sign_info(class_id: int) -> dict:
    """Return dictionary with metadata for given GTSRB Class ID."""
    info = GTSRB_CLASSES.get(class_id, {
        "name": f"Unknown Sign (Class {class_id})",
        "german_name": f"Unbekanntes Schild ({class_id})",
        "category": "Unknown",
        "icon": "❓",
        "description": "No metadata available for this class ID."
    })
    result = dict(info)
    result["class_id"] = class_id
    return result
