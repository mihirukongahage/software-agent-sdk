"""Tests for agent utility functions.

This module tests the prepare_llm_messages and make_llm_completion utility
functions that are used by the agent for message preparation and LLM calls.
"""

from unittest.mock import Mock, patch

import pytest
from pydantic import Field

from openhands.sdk.agent.utils import make_llm_completion, prepare_llm_messages
from openhands.sdk.context.condenser.base import CondenserBase
from openhands.sdk.context.view import View
from openhands.sdk.event import Condensation, MessageEvent
from openhands.sdk.llm import LLM, LLMResponse, Message, TextContent
from openhands.sdk.tool import Action, Observation, ToolDefinition


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock(spec=LLM)
    llm.uses_responses_api.return_value = False
    return llm


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    return [
        MessageEvent(
            source="agent",
            llm_message=Message(
                role="assistant",
                content=[TextContent(text="Hello, how can I help?")],
            ),
        ),
        MessageEvent(
            source="user",
            llm_message=Message(
                role="user",
                content=[TextContent(text="I need help with a task")],
            ),
        ),
        MessageEvent(
            source="agent",
            llm_message=Message(
                role="assistant",
                content=[TextContent(text="I'll help you with that task")],
            ),
        ),
    ]


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        Message(
            role="user",
            content=[TextContent(text="Hello, how can I help?")],
        ),
        Message(
            role="assistant",
            content=[TextContent(text="I need help with a task")],
        ),
        Message(
            role="user",
            content=[TextContent(text="I'll help you with that task")],
        ),
    ]


@pytest.fixture
def mock_condenser():
    """Create a mock condenser for testing."""
    return Mock(spec=CondenserBase)


class MockAgentUtilsAction(Action):
    """Mock action for agent utils testing."""

    param1: str = Field(description="First parameter")


class MockAgentUtilsObservation(Observation):
    """Mock observation for agent utils testing."""

    result: str = Field(description="Result of the action")

    @property
    def to_llm_content(self):
        return [TextContent(text=self.result)]


class MockAgentUtilsTool(
    ToolDefinition[MockAgentUtilsAction, MockAgentUtilsObservation]
):
    """Mock tool definition for agent utils testing."""

    @classmethod
    def create(cls, conv_state=None, **params):
        return [cls(**params)]


@pytest.fixture
def sample_tools():
    """Create sample tool definitions for testing."""
    return [
        MockAgentUtilsTool(
            description="A test tool for agent utils",
            action_type=MockAgentUtilsAction,
            observation_type=MockAgentUtilsObservation,
        )
    ]


# ---------------------------------------------------------------------------
# Tests for prepare_llm_messages
# ---------------------------------------------------------------------------


@patch("openhands.sdk.agent.utils.View.from_events")
@patch("openhands.sdk.event.base.LLMConvertibleEvent.events_to_messages")
def test_prepare_llm_messages_without_condenser(
    mock_events_to_messages, mock_from_events, sample_events, sample_messages
):
    """Test prepare_llm_messages without condenser."""
    # Setup mocks
    mock_view = Mock(spec=View)
    mock_view.events = sample_events
    mock_from_events.return_value = mock_view
    mock_events_to_messages.return_value = sample_messages

    # Call function
    result = prepare_llm_messages(sample_events)

    # Verify results
    assert result == sample_messages
    mock_from_events.assert_called_once_with(sample_events)
    mock_events_to_messages.assert_called_once_with(sample_events)


@patch("openhands.sdk.agent.utils.View.from_events")
@patch("openhands.sdk.event.base.LLMConvertibleEvent.events_to_messages")
def test_prepare_llm_messages_with_additional_messages(
    mock_events_to_messages, mock_from_events, sample_events, sample_messages
):
    """Test prepare_llm_messages with additional messages."""
    # Setup mocks
    mock_view = Mock(spec=View)
    mock_view.events = sample_events
    mock_from_events.return_value = mock_view
    # Create a copy to avoid mutation issues
    mock_events_to_messages.return_value = sample_messages.copy()

    additional_messages = [
        Message(
            role="user",
            content=[TextContent(text="Additional question")],
        )
    ]

    # Call function
    result = prepare_llm_messages(
        sample_events, additional_messages=additional_messages
    )

    # Verify results
    expected_messages = sample_messages + additional_messages
    assert result == expected_messages
    mock_from_events.assert_called_once_with(sample_events)
    mock_events_to_messages.assert_called_once_with(sample_events)


@patch("openhands.sdk.agent.utils.View.from_events")
@patch("openhands.sdk.event.base.LLMConvertibleEvent.events_to_messages")
def test_prepare_llm_messages_with_condenser_returns_view(
    mock_events_to_messages,
    mock_from_events,
    sample_events,
    sample_messages,
    mock_condenser,
):
    """Test prepare_llm_messages with condenser that returns a View."""
    # Setup mocks
    mock_view = Mock(spec=View)
    mock_view.events = sample_events
    mock_from_events.return_value = mock_view

    condensed_events = sample_events[:2]  # Simulate condensation reducing events
    condensed_view = Mock(spec=View)
    condensed_view.events = condensed_events
    mock_condenser.condense.return_value = condensed_view

    condensed_messages = sample_messages[:2]
    mock_events_to_messages.return_value = condensed_messages

    # Call function
    result = prepare_llm_messages(sample_events, condenser=mock_condenser)

    # Verify results
    assert result == condensed_messages
    mock_from_events.assert_called_once_with(sample_events)
    mock_condenser.condense.assert_called_once_with(mock_view, agent_llm=None)
    mock_events_to_messages.assert_called_once_with(condensed_events)


@patch("openhands.sdk.agent.utils.View.from_events")
def test_prepare_llm_messages_with_condenser_returns_condensation(
    mock_from_events, sample_events, mock_condenser
):
    """Test prepare_llm_messages with condenser that returns a Condensation."""
    # Setup mocks
    mock_view = Mock(spec=View)
    mock_view.events = sample_events
    mock_from_events.return_value = mock_view

    condensation = Condensation(
        summary="Test condensation summary",
        llm_response_id="test-response-id",
    )
    mock_condenser.condense.return_value = condensation

    # Call function
    result = prepare_llm_messages(sample_events, condenser=mock_condenser)

    # Verify results
    assert result == condensation
    mock_from_events.assert_called_once_with(sample_events)
    mock_condenser.condense.assert_called_once_with(mock_view, agent_llm=None)


@patch("openhands.sdk.agent.utils.View.from_events")
@patch("openhands.sdk.event.base.LLMConvertibleEvent.events_to_messages")
def test_prepare_llm_messages_empty_events(mock_events_to_messages, mock_from_events):
    """Test prepare_llm_messages with empty events list."""
    # Setup mocks
    mock_view = Mock(spec=View)
    mock_view.events = []
    mock_from_events.return_value = mock_view
    mock_events_to_messages.return_value = []

    # Call function
    result = prepare_llm_messages([])

    # Verify results
    assert result == []
    mock_from_events.assert_called_once_with([])
    mock_events_to_messages.assert_called_once_with([])


# ---------------------------------------------------------------------------
# Tests for make_llm_completion
# ---------------------------------------------------------------------------


def test_make_llm_completion_with_completion_api(mock_llm, sample_messages):
    """Test make_llm_completion using completion API."""
    # Setup mock
    mock_llm.uses_responses_api.return_value = False
    mock_response = Mock(spec=LLMResponse)
    mock_llm.completion.return_value = mock_response

    # Call function
    result = make_llm_completion(mock_llm, sample_messages)

    # Verify results
    assert result == mock_response
    mock_llm.uses_responses_api.assert_called_once()
    mock_llm.completion.assert_called_once_with(
        messages=sample_messages,
        tools=[],
        add_security_risk_prediction=True,
        on_token=None,
    )
    mock_llm.responses.assert_not_called()


def test_make_llm_completion_with_responses_api(mock_llm, sample_messages):
    """Test make_llm_completion using responses API."""
    # Setup mock
    mock_llm.uses_responses_api.return_value = True
    mock_response = Mock(spec=LLMResponse)
    mock_llm.responses.return_value = mock_response

    # Call function
    result = make_llm_completion(mock_llm, sample_messages)

    # Verify results
    assert result == mock_response
    mock_llm.uses_responses_api.assert_called_once()
    mock_llm.responses.assert_called_once_with(
        messages=sample_messages,
        tools=[],
        include=None,
        store=False,
        add_security_risk_prediction=True,
        on_token=None,
    )
    mock_llm.completion.assert_not_called()


def test_make_llm_completion_with_tools_completion_api(
    mock_llm, sample_messages, sample_tools
):
    """Test make_llm_completion with tools using completion API."""
    # Setup mock
    mock_llm.uses_responses_api.return_value = False
    mock_response = Mock(spec=LLMResponse)
    mock_llm.completion.return_value = mock_response

    # Call function
    result = make_llm_completion(mock_llm, sample_messages, tools=sample_tools)

    # Verify results
    assert result == mock_response
    mock_llm.uses_responses_api.assert_called_once()
    mock_llm.completion.assert_called_once_with(
        messages=sample_messages,
        tools=sample_tools,
        add_security_risk_prediction=True,
        on_token=None,
    )


def test_make_llm_completion_with_tools_responses_api(
    mock_llm, sample_messages, sample_tools
):
    """Test make_llm_completion with tools using responses API."""
    # Setup mock
    mock_llm.uses_responses_api.return_value = True
    mock_response = Mock(spec=LLMResponse)
    mock_llm.responses.return_value = mock_response

    # Call function
    result = make_llm_completion(mock_llm, sample_messages, tools=sample_tools)

    # Verify results
    assert result == mock_response
    mock_llm.uses_responses_api.assert_called_once()
    mock_llm.responses.assert_called_once_with(
        messages=sample_messages,
        tools=sample_tools,
        include=None,
        store=False,
        add_security_risk_prediction=True,
        on_token=None,
    )


def test_make_llm_completion_with_none_tools(mock_llm, sample_messages):
    """Test make_llm_completion with None tools parameter."""
    # Setup mock
    mock_llm.uses_responses_api.return_value = False
    mock_response = Mock(spec=LLMResponse)
    mock_llm.completion.return_value = mock_response

    # Call function
    result = make_llm_completion(mock_llm, sample_messages, tools=None)

    # Verify results
    assert result == mock_response
    mock_llm.completion.assert_called_once_with(
        messages=sample_messages,
        tools=[],
        add_security_risk_prediction=True,
        on_token=None,
    )


def test_make_llm_completion_with_empty_tools_list(mock_llm, sample_messages):
    """Test make_llm_completion with empty tools list."""
    # Setup mock
    mock_llm.uses_responses_api.return_value = False
    mock_response = Mock(spec=LLMResponse)
    mock_llm.completion.return_value = mock_response

    # Call function
    result = make_llm_completion(mock_llm, sample_messages, tools=[])

    # Verify results
    assert result == mock_response
    mock_llm.completion.assert_called_once_with(
        messages=sample_messages,
        tools=[],
        add_security_risk_prediction=True,
        on_token=None,
    )


def test_make_llm_completion_empty_messages(mock_llm):
    """Test make_llm_completion with empty messages list."""
    # Setup mock
    mock_llm.uses_responses_api.return_value = False
    mock_response = Mock(spec=LLMResponse)
    mock_llm.completion.return_value = mock_response

    # Call function
    result = make_llm_completion(mock_llm, [])

    # Verify results
    assert result == mock_response
    mock_llm.completion.assert_called_once_with(
        messages=[],
        tools=[],
        add_security_risk_prediction=True,
        on_token=None,
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@patch("openhands.sdk.agent.utils.View.from_events")
@patch("openhands.sdk.event.base.LLMConvertibleEvent.events_to_messages")
def test_prepare_llm_messages_and_make_llm_completion_integration(
    mock_events_to_messages, mock_from_events, sample_events, sample_messages, mock_llm
):
    """Test integration between prepare_llm_messages and make_llm_completion."""
    # Setup mocks for prepare_llm_messages
    mock_view = Mock(spec=View)
    mock_view.events = sample_events
    mock_from_events.return_value = mock_view
    mock_events_to_messages.return_value = sample_messages

    # Setup mocks for make_llm_completion
    mock_llm.uses_responses_api.return_value = False
    mock_response = Mock(spec=LLMResponse)
    mock_llm.completion.return_value = mock_response

    # Call functions in sequence (simulating real usage)
    messages = prepare_llm_messages(sample_events)
    result = make_llm_completion(mock_llm, messages)

    # Verify results
    assert messages == sample_messages
    assert result == mock_response
    mock_llm.completion.assert_called_once_with(
        messages=sample_messages,
        tools=[],
        add_security_risk_prediction=True,
        on_token=None,
    )


def test_make_llm_completion_api_selection():
    """Test that make_llm_completion correctly selects between completion and responses APIs."""  # noqa: E501
    # Test completion API selection
    mock_llm = Mock(spec=LLM)
    mock_llm.uses_responses_api.return_value = False
    mock_response = Mock(spec=LLMResponse)
    mock_llm.completion.return_value = mock_response

    messages = [
        Message(
            role="user",
            content=[TextContent(text="Hello, test message")],
        )
    ]

    result = make_llm_completion(mock_llm, messages)

    assert result == mock_response
    mock_llm.uses_responses_api.assert_called_once()
    mock_llm.completion.assert_called_once_with(
        messages=messages,
        tools=[],
        add_security_risk_prediction=True,
        on_token=None,
    )
    mock_llm.responses.assert_not_called()

    # Reset mocks and test responses API selection
    mock_llm.reset_mock()
    mock_llm.uses_responses_api.return_value = True
    mock_llm.responses.return_value = mock_response

    result = make_llm_completion(mock_llm, messages)

    assert result == mock_response
    mock_llm.uses_responses_api.assert_called_once()
    mock_llm.responses.assert_called_once_with(
        messages=messages,
        tools=[],
        include=None,
        store=False,
        add_security_risk_prediction=True,
        on_token=None,
    )
    mock_llm.completion.assert_not_called()


# ---------------------------------------------------------------------------
# Tests without mocks to ensure actual code paths are covered
# ---------------------------------------------------------------------------


def test_prepare_llm_messages_real_events():
    """Test prepare_llm_messages with real events (no mocks)."""
    # Create real events
    events = [
        MessageEvent(
            source="user",
            llm_message=Message(
                role="user",
                content=[TextContent(text="Hello")],
            ),
        ),
        MessageEvent(
            source="agent",
            llm_message=Message(
                role="assistant",
                content=[TextContent(text="Hi there!")],
            ),
        ),
    ]

    # Call function without mocks
    result = prepare_llm_messages(events)

    # Verify results
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[1].role == "assistant"


def test_prepare_llm_messages_real_events_with_additional_messages():
    """Test prepare_llm_messages with real events and additional messages."""
    events = [
        MessageEvent(
            source="user",
            llm_message=Message(
                role="user",
                content=[TextContent(text="Hello")],
            ),
        ),
    ]

    additional = [
        Message(
            role="user",
            content=[TextContent(text="Follow-up question")],
        )
    ]

    result = prepare_llm_messages(events, additional_messages=additional)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[1].role == "user"


def test_prepare_llm_messages_real_condenser_returns_view():
    """Test prepare_llm_messages with a real condenser that returns a View."""
    events = [
        MessageEvent(
            source="user",
            llm_message=Message(
                role="user",
                content=[TextContent(text="Hello")],
            ),
        ),
    ]

    # Create a mock condenser that returns a View
    mock_condenser = Mock(spec=CondenserBase)
    condensed_view = View.from_events(events)
    mock_condenser.condense.return_value = condensed_view

    result = prepare_llm_messages(events, condenser=mock_condenser)

    assert isinstance(result, list)
    mock_condenser.condense.assert_called_once()


def test_prepare_llm_messages_real_condenser_returns_condensation():
    """Test prepare_llm_messages with a condenser that returns a Condensation."""
    events = [
        MessageEvent(
            source="user",
            llm_message=Message(
                role="user",
                content=[TextContent(text="Hello")],
            ),
        ),
    ]

    # Create a mock condenser that returns a Condensation
    mock_condenser = Mock(spec=CondenserBase)
    condensation = Condensation(
        summary="Condensed summary",
        llm_response_id="test-id",
    )
    mock_condenser.condense.return_value = condensation

    result = prepare_llm_messages(events, condenser=mock_condenser)

    assert isinstance(result, Condensation)
    assert result.summary == "Condensed summary"


def test_prepare_llm_messages_with_llm_parameter():
    """Test prepare_llm_messages passes llm to condenser."""
    events = [
        MessageEvent(
            source="user",
            llm_message=Message(
                role="user",
                content=[TextContent(text="Hello")],
            ),
        ),
    ]

    mock_condenser = Mock(spec=CondenserBase)
    mock_condenser.condense.return_value = View.from_events(events)

    mock_llm = Mock(spec=LLM)

    prepare_llm_messages(events, condenser=mock_condenser, llm=mock_llm)

    # Verify llm was passed to condenser
    mock_condenser.condense.assert_called_once()
    call_kwargs = mock_condenser.condense.call_args[1]
    assert call_kwargs["agent_llm"] == mock_llm


def test_make_llm_completion_with_on_token_callback():
    """Test make_llm_completion with on_token callback."""
    mock_llm = Mock(spec=LLM)
    mock_llm.uses_responses_api.return_value = False
    mock_response = Mock(spec=LLMResponse)
    mock_llm.completion.return_value = mock_response

    messages = [
        Message(
            role="user",
            content=[TextContent(text="Hello")],
        )
    ]

    def token_callback(token):
        pass

    result = make_llm_completion(mock_llm, messages, on_token=token_callback)

    assert result == mock_response
    mock_llm.completion.assert_called_once_with(
        messages=messages,
        tools=[],
        add_security_risk_prediction=True,
        on_token=token_callback,
    )


def test_make_llm_completion_responses_api_with_on_token():
    """Test make_llm_completion with responses API and on_token callback."""
    mock_llm = Mock(spec=LLM)
    mock_llm.uses_responses_api.return_value = True
    mock_response = Mock(spec=LLMResponse)
    mock_llm.responses.return_value = mock_response

    messages = [
        Message(
            role="user",
            content=[TextContent(text="Hello")],
        )
    ]

    def token_callback(token):
        pass

    result = make_llm_completion(mock_llm, messages, on_token=token_callback)

    assert result == mock_response
    mock_llm.responses.assert_called_once_with(
        messages=messages,
        tools=[],
        include=None,
        store=False,
        add_security_risk_prediction=True,
        on_token=token_callback,
    )
