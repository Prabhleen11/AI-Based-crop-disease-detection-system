"""
treatment_recommendations.py

Maps predicted disease classes to short, farmer-friendly treatment suggestions.
Extend this dictionary with the actual class names from your dataset.
"""

TREATMENT_MAP = {
    "Healthy": "No disease detected. Continue regular monitoring and balanced fertilization.",
    "Early_Blight": "Remove and destroy affected leaves. Apply a copper-based or chlorothalonil "
                    "fungicide. Avoid overhead watering to reduce leaf moisture.",
    "Late_Blight": "Apply a systemic fungicide (e.g. metalaxyl-based). Improve field drainage and "
                   "spacing between plants to increase airflow.",
    "Bacterial_Spot": "Use copper-based bactericides. Avoid working in fields when leaves are wet "
                       "to prevent spreading the bacteria.",
}


def get_recommendation(predicted_class):
    """
    Returns a treatment suggestion string for a given predicted class name.
    Falls back to a generic message if the class is not in the map.
    """
    return TREATMENT_MAP.get(
        predicted_class,
        "No specific recommendation available. Consult a local agricultural expert for guidance.",
    )
