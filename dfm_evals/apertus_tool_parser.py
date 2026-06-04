from __future__ import annotations

import json
from collections.abc import Sequence

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser, ToolParserManager


@ToolParserManager.register_module("apertus")
class ApertusToolParser(ToolParser):
    """Parse Apertus native tool calls.

    Apertus serializes calls as:
    <|tools_prefix|>[{"tool_name": {"arg": "value"}}]<|tools_suffix|>
    """

    tool_call_start_token = "<|tools_prefix|>"
    tool_call_end_token = "<|tools_suffix|>"
    tool_call_regex = re.compile(
        r"<\|tools_prefix\|>(.*?)<\|tools_suffix\|>|<\|tools_prefix\|>(.*)",
        re.DOTALL,
    )

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        token_ids: Sequence[int] | None = None,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        match = self.tool_call_regex.search(model_output)
        if match is None:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        raw_calls = (match.group(1) or match.group(2) or "").strip()
        content = model_output[: match.start()].strip() or None
        try:
            parsed_calls = json.loads(raw_calls)
        except json.JSONDecodeError:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        if isinstance(parsed_calls, dict):
            parsed_calls = [parsed_calls]
        if not isinstance(parsed_calls, list):
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[ToolCall] = []
        for call in parsed_calls:
            if not isinstance(call, dict) or len(call) != 1:
                continue
            name, arguments = next(iter(call.items()))
            if not isinstance(name, str) or not name:
                continue
            tool_calls.append(
                ToolCall(
                    id=make_tool_call_id(),
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=json.dumps(
                            arguments if arguments is not None else {},
                            ensure_ascii=False,
                        ),
                    ),
                )
            )

        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content=content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if self.tool_call_start_token not in current_text and delta_text:
            return DeltaMessage(content=delta_text)
        return None
