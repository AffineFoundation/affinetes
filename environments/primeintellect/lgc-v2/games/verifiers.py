"""
Verifier Registry

Register all task verifiers here.
"""

from games.dyck_language.verifier import DyckLanguageVerifier

# Add your verifiers here:
# from games.your_task.verifier import YourTaskVerifier

verifier_classes = {
    "dyck_language": DyckLanguageVerifier,
    # Add your verifiers here:
    # "your_task": YourTaskVerifier,
}
