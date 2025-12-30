"""
Verifier Registry

Register all task verifiers here.
"""

from games.dyck_language.verifier import DyckLanguageVerifier
from games.game_of_24.verifier import GameOf24Verifier
from games.operation.verifier import OperationVerifier
from games.cryptarithm.verifier import CryptarithmVerifier

# Add your verifiers here:
# from games.your_task.verifier import YourTaskVerifier

verifier_classes = {
    "dyck_language": DyckLanguageVerifier,
    "game_of_24": GameOf24Verifier,
    "operation": OperationVerifier,
    "cryptarithm": CryptarithmVerifier,
    # Add your verifiers here:
    # "your_task": YourTaskVerifier,
}
