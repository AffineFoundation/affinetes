"""Adapter implementations.

One subpackage per port (teachers, normalizers, ...). Adapters do all the IO,
provider-specific parsing, and any framework binding. They are wired up behind
ports in ``application`` use cases.
"""
