"""Application use cases.

This layer orchestrates ports to fulfil a single business intent. No HTTP, no
ORM, no LLM SDKs — only domain models and Protocol-typed dependencies passed
in by the caller (workers, CLI, API routers).
"""
