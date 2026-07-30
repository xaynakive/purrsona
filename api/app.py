from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
personality_model = joblib.load("personality_model.pkl")
encoders = joblib.load("feature_encoders.pkl")
target_encoders = joblib.load("target_encoders.pkl")

# Fashion recommendations
personality_fashion = {
    'Royal': {
        'outfit': 'Golden Cape',
        'accessory': 'Crystal Crown',
        'travel': 'Gold Vanity Case',
        'color': 'Cream White',
        'vibe': 'Regal and dignified — you were born to rule 👑',
        'emoji': '👑'
    },
    'Bossy': {
        'outfit': 'Velvet Coat',
        'accessory': 'Pearl Necklace',
        'travel': 'Mini Briefcase',
        'color': 'Midnight Black',
        'vibe': 'Commanding and confident — no one questions you 🖤',
        'emoji': '😼'
    },
    'Chaotic': {
        'outfit': 'Denim Jacket',
        'accessory': 'Star Pins',
        'travel': 'Cherry Red Backpack',
        'color': 'Cherry Red',
        'vibe': 'Unpredictable and energetic — zoomies at 3am 🌀',
        'emoji': '⚡'
    },
    'Dreamy': {
        'outfit': 'Lavender Dress',
        'accessory': 'Flower Garland',
        'travel': 'Lavender Tote Bag',
        'color': 'Lavender',
        'vibe': 'Soft and whimsical — living in a dream world 🌸',
        'emoji': '🌸'
    },
    'Shy': {
        'outfit': 'Fluffy Hoodie',
        'accessory': 'Satin Bow',
        'travel': 'Rattan Bag',
        'color': 'Icy Blue',
        'vibe': 'Gentle and reserved — selective with trust 🩵',
        'emoji': '🩵'
    }
}

categorical_cols = ['breed','fur_color','eye_color','size','energy_level',
                    'vocalization','affection_level','coat_length',
                    'indoor_outdoor','grooming_needs','social_with_humans','social_with_cats']

class CatInput(BaseModel):
    breed: str
    fur_color: str
    eye_color: str
    size: str
    energy_level: str
    vocalization: str
    affection_level: str
    coat_length: str
    age_months: int
    weight_kg: float
    indoor_outdoor: str
    playfulness_score: int
    independence_score: int
    grooming_needs: str
    social_with_humans: str
    social_with_cats: str

@app.post("/predict")
def predict(cat: CatInput):
    sample = pd.DataFrame([cat.dict()])
    
    for col in categorical_cols:
        sample[col] = encoders[col].transform(sample[col])
    
    personality = target_encoders['personality'].inverse_transform(
        personality_model.predict(sample)
    )[0]
    
    fashion = personality_fashion[personality]
    
    return {
        "personality": personality,
        "emoji": fashion["emoji"],
        "vibe": fashion["vibe"],
        "outfit": fashion["outfit"],
        "accessory": fashion["accessory"],
        "travel": fashion["travel"],
        "color": fashion["color"]
    }

@app.get("/")
def root():
    return {"message": "PurrSona API is running! 🐱"}
