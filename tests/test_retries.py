from unittest.mock import MagicMock

import pytest

from polylogue.ingestion import DriveAuthError, DriveClient


def test_drive_retry_success():
    """Test that it succeeds immediately without retry."""
    client = DriveClient(retries=2, retry_base=0.01)
    mock_func = MagicMock(return_value="success")

    result = client._call_with_retry(mock_func)

    assert result == "success"
    assert mock_func.call_count == 1


def test_drive_retry_eventual_success():
    """Test that it retries and eventually succeeds."""
    client = DriveClient(retries=2, retry_base=0.01)
    mock_func = MagicMock(side_effect=[Exception("fail"), "success"])

    result = client._call_with_retry(mock_func)

    assert result == "success"
    assert mock_func.call_count == 2


def test_drive_retry_failure():
    """Test that it fails after max retries."""
    client = DriveClient(retries=2, retry_base=0.01)
    mock_func = MagicMock(side_effect=Exception("fail"))

    with pytest.raises(Exception, match="fail"):
        client._call_with_retry(mock_func)

    # Initial call + 2 retries = 3 calls
    assert mock_func.call_count == 3


def test_drive_no_retry_on_auth_error():
    """Test that it fails immediately on DriveAuthError."""
    client = DriveClient(retries=2, retry_base=0.01)
    mock_func = MagicMock(side_effect=DriveAuthError("auth fail"))

    with pytest.raises(DriveAuthError):
        client._call_with_retry(mock_func)

    assert mock_func.call_count == 1
