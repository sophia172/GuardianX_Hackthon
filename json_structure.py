from pydantic import BaseModel


class DecisionInfo(BaseModel):
    image_description: str
    weapon_detect: bool
    danger_detect: bool
    emergency: bool


decision_prompts = {
    "image_description": "Provide a brief description of the image.",
    "weapon_detect": "Is there a knife in the image?",
    "danger_detect": "There is a knife in the image, Is this a dangerous scenario for the subject?",
    "emergency": "Is it a situation for emergency call?",
}


class QuestionInfo(BaseModel):
    injury: bool
    location: bool
    situation: bool
    people: bool
    armed: bool
    describe_people: bool


question_prompts = {
    "injury": "Anyone injured?",
    "location": "Is it asking for the location?",
    "situation": "Is it asking for the situation?",
    "people": "Is it asking for the people on the scene?",
    "armed": "Is it asking for weapons?",
    "describe_people": "Is it asking you to describe people around you",
}


class SituationInfo(BaseModel):
    injury: str
    situation: str
    people: str
    armed: str
    describe_people: str


situation_prompts = {
    "situation": "Describe the emergency situation based on the image in a few sentences.",
    "people": "How many people in the scene? Answer in one sentence.",
    "armed": "Is anyone armed with a weapon? Answer in one sentence.",
    "injury": "Any people injured on the scene? Answer in one sentence.",
    "describe_people": "Describe everyone shown in the image.",
}
