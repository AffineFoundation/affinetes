"""Data class for task challenges"""
import json


class Data:
    """Data object for task challenges"""

    def __init__(self, question: str, answer: str, difficulty: int = 1, metadata: dict = None, **kwargs):
        self.question = question
        self.answer = answer
        self.difficulty = difficulty
        self.metadata = metadata or {}
        self.gpt_response = ""

    def to_json(self):
        """Convert to JSON dict"""
        return {
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "gpt_response": self.gpt_response,
        }

    def to_json_str(self):
        """Convert to JSON string"""
        return json.dumps(self.to_json(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_dict):
        """Create from JSON dict"""
        instance = cls(
            question=json_dict.get("question", ""),
            answer=json_dict.get("answer", ""),
            difficulty=json_dict.get("difficulty", 1),
            metadata=json_dict.get("metadata", {})
        )
        if "gpt_response" in json_dict:
            instance.gpt_response = json_dict["gpt_response"]
        return instance

    @classmethod
    def from_json_str(cls, json_str):
        """Create from JSON string"""
        json_data = json.loads(json_str)
        return cls.from_json(json_data)
