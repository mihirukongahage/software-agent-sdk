"""Tests for CustomBrowserUseServer in server.py."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.tools.browser_use.server import CustomBrowserUseServer


class TestGetStorage:
    """Tests for _get_storage method."""

    @pytest.fixture
    def server(self):
        """Create a CustomBrowserUseServer instance with mocked parent."""
        with patch(
            "openhands.tools.browser_use.server.LogSafeBrowserUseServer.__init__",
            return_value=None,
        ):
            server = CustomBrowserUseServer()
            server.browser_session = None
            return server

    @pytest.mark.asyncio
    async def test_get_storage_no_browser_session(self, server):
        """Test _get_storage returns error when no browser session is active."""
        server.browser_session = None
        result = await server._get_storage()
        assert result == "Error: No browser session active"

    @pytest.mark.asyncio
    async def test_get_storage_success(self, server):
        """Test _get_storage returns JSON storage state on success."""
        mock_session = MagicMock()
        mock_storage_state = {
            "cookies": [{"name": "test_cookie", "value": "test_value"}],
            "origins": [{"origin": "https://example.com", "localStorage": []}],
        }
        mock_session._cdp_get_storage_state = AsyncMock(return_value=mock_storage_state)
        server.browser_session = mock_session

        result = await server._get_storage()

        expected = json.dumps(mock_storage_state, indent=2)
        assert result == expected
        mock_session._cdp_get_storage_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_storage_exception(self, server):
        """Test _get_storage handles exceptions properly."""
        mock_session = MagicMock()
        mock_session._cdp_get_storage_state = AsyncMock(
            side_effect=Exception("CDP error")
        )
        server.browser_session = mock_session

        result = await server._get_storage()

        assert "Error getting storage state" in result
        assert "CDP error" in result


class TestSetStorage:
    """Tests for _set_storage method."""

    @pytest.fixture
    def server(self):
        """Create a CustomBrowserUseServer instance with mocked parent."""
        with patch(
            "openhands.tools.browser_use.server.LogSafeBrowserUseServer.__init__",
            return_value=None,
        ):
            server = CustomBrowserUseServer()
            server.browser_session = None
            return server

    @pytest.mark.asyncio
    async def test_set_storage_no_browser_session(self, server):
        """Test _set_storage returns error when no browser session is active."""
        server.browser_session = None
        result = await server._set_storage({})
        assert result == "Error: No browser session active"

    @pytest.mark.asyncio
    async def test_set_storage_with_cookies_only(self, server):
        """Test _set_storage sets cookies successfully."""
        mock_session = MagicMock()
        mock_session._cdp_set_cookies = AsyncMock()
        server.browser_session = mock_session

        storage_state = {
            "cookies": [{"name": "test_cookie", "value": "test_value"}],
        }
        result = await server._set_storage(storage_state)

        assert result == "Storage set successfully"
        mock_session._cdp_set_cookies.assert_called_once_with(storage_state["cookies"])

    @pytest.mark.asyncio
    async def test_set_storage_with_origins_and_localstorage(self, server):
        """Test _set_storage sets localStorage items."""
        mock_session = MagicMock()
        mock_session._cdp_set_cookies = AsyncMock()

        mock_cdp_client = MagicMock()
        mock_dom_storage = MagicMock()
        mock_dom_storage.enable = AsyncMock()
        mock_dom_storage.disable = AsyncMock()
        mock_dom_storage.setDOMStorageItem = AsyncMock()
        mock_cdp_client.send.DOMStorage = mock_dom_storage

        mock_cdp_session = MagicMock()
        mock_cdp_session.cdp_client = mock_cdp_client
        mock_cdp_session.session_id = "test_session_id"

        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)
        server.browser_session = mock_session

        storage_state = {
            "cookies": [],
            "origins": [
                {
                    "origin": "https://example.com",
                    "localStorage": [{"key": "test_key", "value": "test_value"}],
                }
            ],
        }
        result = await server._set_storage(storage_state)

        assert result == "Storage set successfully"
        mock_dom_storage.enable.assert_called_once()
        mock_dom_storage.setDOMStorageItem.assert_called()
        mock_dom_storage.disable.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_storage_with_origins_and_sessionstorage(self, server):
        """Test _set_storage sets sessionStorage items."""
        mock_session = MagicMock()
        mock_session._cdp_set_cookies = AsyncMock()

        mock_cdp_client = MagicMock()
        mock_dom_storage = MagicMock()
        mock_dom_storage.enable = AsyncMock()
        mock_dom_storage.disable = AsyncMock()
        mock_dom_storage.setDOMStorageItem = AsyncMock()
        mock_cdp_client.send.DOMStorage = mock_dom_storage

        mock_cdp_session = MagicMock()
        mock_cdp_session.cdp_client = mock_cdp_client
        mock_cdp_session.session_id = "test_session_id"

        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)
        server.browser_session = mock_session

        storage_state = {
            "cookies": [],
            "origins": [
                {
                    "origin": "https://example.com",
                    "sessionStorage": [{"key": "session_key", "value": "session_value"}],
                }
            ],
        }
        result = await server._set_storage(storage_state)

        assert result == "Storage set successfully"
        mock_dom_storage.setDOMStorageItem.assert_called()

    @pytest.mark.asyncio
    async def test_set_storage_with_name_instead_of_key(self, server):
        """Test _set_storage handles 'name' field instead of 'key'."""
        mock_session = MagicMock()
        mock_session._cdp_set_cookies = AsyncMock()

        mock_cdp_client = MagicMock()
        mock_dom_storage = MagicMock()
        mock_dom_storage.enable = AsyncMock()
        mock_dom_storage.disable = AsyncMock()
        mock_dom_storage.setDOMStorageItem = AsyncMock()
        mock_cdp_client.send.DOMStorage = mock_dom_storage

        mock_cdp_session = MagicMock()
        mock_cdp_session.cdp_client = mock_cdp_client
        mock_cdp_session.session_id = "test_session_id"

        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)
        server.browser_session = mock_session

        storage_state = {
            "origins": [
                {
                    "origin": "https://example.com",
                    "localStorage": [{"name": "test_name", "value": "test_value"}],
                }
            ],
        }
        result = await server._set_storage(storage_state)

        assert result == "Storage set successfully"

    @pytest.mark.asyncio
    async def test_set_storage_skips_items_without_key_or_name(self, server):
        """Test _set_storage skips items without key or name."""
        mock_session = MagicMock()
        mock_session._cdp_set_cookies = AsyncMock()

        mock_cdp_client = MagicMock()
        mock_dom_storage = MagicMock()
        mock_dom_storage.enable = AsyncMock()
        mock_dom_storage.disable = AsyncMock()
        mock_dom_storage.setDOMStorageItem = AsyncMock()
        mock_cdp_client.send.DOMStorage = mock_dom_storage

        mock_cdp_session = MagicMock()
        mock_cdp_session.cdp_client = mock_cdp_client
        mock_cdp_session.session_id = "test_session_id"

        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)
        server.browser_session = mock_session

        storage_state = {
            "origins": [
                {
                    "origin": "https://example.com",
                    "localStorage": [{"value": "no_key_value"}],
                    "sessionStorage": [{"value": "no_key_session_value"}],
                }
            ],
        }
        result = await server._set_storage(storage_state)

        assert result == "Storage set successfully"
        # setDOMStorageItem should not be called since items have no key
        mock_dom_storage.setDOMStorageItem.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_storage_skips_origins_without_origin_field(self, server):
        """Test _set_storage skips origin entries without origin field."""
        mock_session = MagicMock()
        mock_session._cdp_set_cookies = AsyncMock()

        mock_cdp_client = MagicMock()
        mock_dom_storage = MagicMock()
        mock_dom_storage.enable = AsyncMock()
        mock_dom_storage.disable = AsyncMock()
        mock_dom_storage.setDOMStorageItem = AsyncMock()
        mock_cdp_client.send.DOMStorage = mock_dom_storage

        mock_cdp_session = MagicMock()
        mock_cdp_session.cdp_client = mock_cdp_client
        mock_cdp_session.session_id = "test_session_id"

        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)
        server.browser_session = mock_session

        storage_state = {
            "origins": [
                {
                    "localStorage": [{"key": "test_key", "value": "test_value"}],
                }
            ],
        }
        result = await server._set_storage(storage_state)

        assert result == "Storage set successfully"
        # setDOMStorageItem should not be called since origin is missing
        mock_dom_storage.setDOMStorageItem.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_storage_exception(self, server):
        """Test _set_storage handles exceptions properly."""
        mock_session = MagicMock()
        mock_session._cdp_set_cookies = AsyncMock(side_effect=Exception("Set error"))
        server.browser_session = mock_session

        storage_state = {"cookies": [{"name": "test", "value": "value"}]}
        result = await server._set_storage(storage_state)

        assert "Error setting storage state" in result
        assert "Set error" in result


class TestGetContent:
    """Tests for _get_content method."""

    @pytest.fixture
    def server(self):
        """Create a CustomBrowserUseServer instance with mocked parent."""
        with patch(
            "openhands.tools.browser_use.server.LogSafeBrowserUseServer.__init__",
            return_value=None,
        ):
            server = CustomBrowserUseServer()
            server.browser_session = None
            return server

    @pytest.mark.asyncio
    async def test_get_content_no_browser_session(self, server):
        """Test _get_content returns error when no browser session is active."""
        server.browser_session = None
        result = await server._get_content()
        assert result == "Error: No browser session active"

    @pytest.mark.asyncio
    async def test_get_content_success(self, server):
        """Test _get_content returns formatted content on success."""
        mock_session = MagicMock()
        mock_session.get_current_page_url = AsyncMock(
            return_value="https://example.com"
        )
        server.browser_session = mock_session

        content = "Test content"
        content_stats = {
            "original_html_chars": 1000,
            "initial_markdown_chars": 500,
            "final_filtered_chars": 400,
            "filtered_chars_removed": 100,
        }

        with patch(
            "openhands.tools.browser_use.server.extract_clean_markdown",
            new_callable=AsyncMock,
            return_value=(content, content_stats),
        ):
            result = await server._get_content()

        assert "<url>" in result
        assert "https://example.com" in result
        assert "<content>" in result
        assert "Test content" in result
        assert "<content_stats>" in result

    @pytest.mark.asyncio
    async def test_get_content_with_extract_links(self, server):
        """Test _get_content with extract_links=True."""
        mock_session = MagicMock()
        mock_session.get_current_page_url = AsyncMock(
            return_value="https://example.com"
        )
        server.browser_session = mock_session

        content = "Content with [link](https://example.com)"
        content_stats = {
            "original_html_chars": 1000,
            "initial_markdown_chars": 500,
            "final_filtered_chars": 400,
            "filtered_chars_removed": 100,
        }

        with patch(
            "openhands.tools.browser_use.server.extract_clean_markdown",
            new_callable=AsyncMock,
            return_value=(content, content_stats),
        ) as mock_extract:
            result = await server._get_content(extract_links=True)

        mock_extract.assert_called_once_with(
            browser_session=mock_session, extract_links=True
        )
        assert "link" in result

    @pytest.mark.asyncio
    async def test_get_content_with_start_from_char(self, server):
        """Test _get_content with start_from_char parameter."""
        mock_session = MagicMock()
        mock_session.get_current_page_url = AsyncMock(
            return_value="https://example.com"
        )
        server.browser_session = mock_session

        content = "0123456789" * 100  # 1000 chars
        content_stats = {
            "original_html_chars": 2000,
            "initial_markdown_chars": 1500,
            "final_filtered_chars": 1000,
            "filtered_chars_removed": 500,
        }

        with patch(
            "openhands.tools.browser_use.server.extract_clean_markdown",
            new_callable=AsyncMock,
            return_value=(content, content_stats),
        ):
            result = await server._get_content(start_from_char=100)

        assert "started from char 100" in result

    @pytest.mark.asyncio
    async def test_get_content_start_from_char_exceeds_length(self, server):
        """Test _get_content when start_from_char exceeds content length."""
        mock_session = MagicMock()
        server.browser_session = mock_session

        content = "Short content"
        content_stats = {
            "original_html_chars": 100,
            "initial_markdown_chars": 50,
            "final_filtered_chars": 13,
            "filtered_chars_removed": 37,
        }

        with patch(
            "openhands.tools.browser_use.server.extract_clean_markdown",
            new_callable=AsyncMock,
            return_value=(content, content_stats),
        ):
            result = await server._get_content(start_from_char=1000)

        assert "start_from_char (1000) exceeds content length" in result

    @pytest.mark.asyncio
    async def test_get_content_extraction_exception(self, server):
        """Test _get_content handles extraction exceptions."""
        mock_session = MagicMock()
        server.browser_session = mock_session

        with patch(
            "openhands.tools.browser_use.server.extract_clean_markdown",
            new_callable=AsyncMock,
            side_effect=ValueError("Extraction failed"),
        ):
            result = await server._get_content()

        assert "Could not extract clean markdown" in result
        assert "ValueError" in result

    @pytest.mark.asyncio
    async def test_get_content_truncation(self, server):
        """Test _get_content truncates long content."""
        mock_session = MagicMock()
        mock_session.get_current_page_url = AsyncMock(
            return_value="https://example.com"
        )
        server.browser_session = mock_session

        # Create content longer than MAX_CHAR_LIMIT (30000)
        content = "A" * 35000
        content_stats = {
            "original_html_chars": 50000,
            "initial_markdown_chars": 40000,
            "final_filtered_chars": 35000,
            "filtered_chars_removed": 5000,
        }

        with patch(
            "openhands.tools.browser_use.server.extract_clean_markdown",
            new_callable=AsyncMock,
            return_value=(content, content_stats),
        ):
            result = await server._get_content()

        assert "truncated" in result
        assert "start_from_char=" in result

    @pytest.mark.asyncio
    async def test_get_content_truncation_at_paragraph_break(self, server):
        """Test _get_content truncates at paragraph break when possible."""
        mock_session = MagicMock()
        mock_session.get_current_page_url = AsyncMock(
            return_value="https://example.com"
        )
        server.browser_session = mock_session

        # Create content with paragraph break near truncation point
        content = "A" * 29600 + "\n\n" + "B" * 5000
        content_stats = {
            "original_html_chars": 50000,
            "initial_markdown_chars": 40000,
            "final_filtered_chars": len(content),
            "filtered_chars_removed": 5000,
        }

        with patch(
            "openhands.tools.browser_use.server.extract_clean_markdown",
            new_callable=AsyncMock,
            return_value=(content, content_stats),
        ):
            result = await server._get_content()

        assert "truncated" in result

    @pytest.mark.asyncio
    async def test_get_content_truncation_at_sentence_break(self, server):
        """Test _get_content truncates at sentence break when no paragraph break."""
        mock_session = MagicMock()
        mock_session.get_current_page_url = AsyncMock(
            return_value="https://example.com"
        )
        server.browser_session = mock_session

        # Create content with sentence break near truncation point but no paragraph
        content = "A" * 29850 + ". " + "B" * 5000
        content_stats = {
            "original_html_chars": 50000,
            "initial_markdown_chars": 40000,
            "final_filtered_chars": len(content),
            "filtered_chars_removed": 5000,
        }

        with patch(
            "openhands.tools.browser_use.server.extract_clean_markdown",
            new_callable=AsyncMock,
            return_value=(content, content_stats),
        ):
            result = await server._get_content()

        assert "truncated" in result

    @pytest.mark.asyncio
    async def test_get_content_with_filtered_chars(self, server):
        """Test _get_content shows filtered chars info when chars were filtered."""
        mock_session = MagicMock()
        mock_session.get_current_page_url = AsyncMock(
            return_value="https://example.com"
        )
        server.browser_session = mock_session

        content = "Clean content"
        content_stats = {
            "original_html_chars": 1000,
            "initial_markdown_chars": 500,
            "final_filtered_chars": 13,
            "filtered_chars_removed": 487,
        }

        with patch(
            "openhands.tools.browser_use.server.extract_clean_markdown",
            new_callable=AsyncMock,
            return_value=(content, content_stats),
        ):
            result = await server._get_content()

        assert "filtered 487 chars of noise" in result
