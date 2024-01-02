"""
Helper functions for creating and reading UTC datetime strings.
"""
from datetime import datetime, timezone


def utc_datetime_to_str(utc_datetime: datetime) -> str:
    """
    Construct and return a 20-char string representing the UTC datetime in the
    format "%Y-%m-%dT%H_%M_%SZ".
    """
    assert utc_datetime.tzinfo is None \
        or utc_datetime.tzinfo == timezone.utc
    return utc_datetime.strftime("%Y-%m-%dT%H_%M_%SZ")


def current_utc_datetime_str() -> str:
    """
    Construct and return a 20-char string representing the current UTC datetime
    in the format "%Y-%m-%dT%H_%M_%SZ".
    """
    return datetime.utcnow().strftime("%Y-%m-%dT%H_%M_%SZ")


def utc_datetime_from_str_prefix(s: str) -> datetime:
    """
    Construct and return a UTC datetime object from a string prefix.

    Args:
        s: a string prefixed with a UTC datetime in the format
            "%Y-%m-%dT%H_%M_%SZ" as output by `current_utc_datetime_str`.

    Returns:
        UTC datetime object
    """
    prefix: str = s[:20]
    assert prefix[19] == "Z", "Datetime must be UTC!"
    return datetime.strptime(
        prefix,
        "%Y-%m-%dT%H_%M_%SZ",
    ).replace(tzinfo=timezone.utc)
