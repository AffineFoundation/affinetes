"""
Verifier Registry

Register all task verifiers here.
"""

from games.dyck_language.verifier import DyckLanguageVerifier
from games.dyck_language2.verifier import DyckLanguage2Verifier
from games.game_of_24.verifier import GameOf24Verifier
from games.operation.verifier import OperationVerifier
from games.cryptarithm.verifier import CryptarithmVerifier
from games.boolean_expressions.verifier import BooleanExpressionsVerifier

# Add your verifiers here:
# from games.your_task.verifier import YourTaskVerifier

verifier_classes = {
    "dyck_language": DyckLanguageVerifier,
    "dyck_language2": DyckLanguage2Verifier,
    "game_of_24": GameOf24Verifier,
    "operation": OperationVerifier,
    "cryptarithm": CryptarithmVerifier,
    "boolean_expressions": BooleanExpressionsVerifier,
    # Add your verifiers here:
    # "your_task": YourTaskVerifier,
}
