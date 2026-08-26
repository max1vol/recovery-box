"""Errors raised at the local-data and federated-protocol boundaries."""


class FederationError(ValueError):
    """Base class for rejected federation input."""


class SanitizedDataError(FederationError):
    """A local JSONL feature record is malformed or outside the approved schema."""


class ModelSchemaError(FederationError):
    """Model parameters or feature schema are incompatible with this client."""


class FederationConfigError(FederationError):
    """Federation configuration is missing, ambiguous, or unsafe to interpret."""
