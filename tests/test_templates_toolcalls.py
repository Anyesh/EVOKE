from evoke.templates import parse_qwen_response


class TestParseJsonToolCall:
    def test_json_body_parses(self):
        raw = '<tool_call>\n{"name": "write", "arguments": {"path": "a.txt", "content": "hi"}}\n</tool_call>'
        parsed = parse_qwen_response(raw)
        assert parsed.finish_reason == "tool_calls"
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].name == "write"
        assert parsed.tool_calls[0].arguments == {"path": "a.txt", "content": "hi"}

    def test_json_string_arguments_decoded(self):
        raw = '<tool_call>{"name": "ls", "arguments": "{\\"path\\": \\".\\"}"}</tool_call>'
        parsed = parse_qwen_response(raw)
        assert parsed.tool_calls[0].arguments == {"path": "."}


class TestParseFunctionXmlToolCall:
    def test_qwen35_function_xml_parses(self):
        raw = (
            "<tool_call>\n<function=write>\n<parameter=path>\nhello.txt\n</parameter>\n"
            "<parameter=content>\nhi\n</parameter>\n</function>\n</tool_call>"
        )
        parsed = parse_qwen_response(raw)
        assert parsed.finish_reason == "tool_calls"
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].name == "write"
        assert parsed.tool_calls[0].arguments == {"path": "hello.txt", "content": "hi"}
        assert parsed.content == ""

    def test_multiline_parameter_value_preserved(self):
        body = "line one\nline two\n  indented"
        raw = (
            "<tool_call>\n<function=write>\n<parameter=path>\napp.py\n</parameter>\n"
            f"<parameter=content>\n{body}\n</parameter>\n</function>\n</tool_call>"
        )
        parsed = parse_qwen_response(raw)
        assert parsed.tool_calls[0].arguments["content"] == body

    def test_json_typed_parameter_values_coerced_via_schema(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "todowrite",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "todos": {"type": "array"},
                            "merge": {"type": "boolean"},
                        },
                    },
                },
            }
        ]
        raw = (
            "<tool_call>\n<function=todowrite>\n<parameter=todos>\n"
            '[{"id": "1", "content": "x"}]\n</parameter>\n'
            "<parameter=merge>\ntrue\n</parameter>\n</function>\n</tool_call>"
        )
        parsed = parse_qwen_response(raw, tools=tools)
        args = parsed.tool_calls[0].arguments
        assert args["todos"] == [{"id": "1", "content": "x"}]
        assert args["merge"] is True

    def test_string_param_that_looks_like_json_stays_string(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                },
            }
        ]
        raw = (
            "<tool_call>\n<function=write>\n<parameter=path>\nconf.json\n</parameter>\n"
            '<parameter=content>\n{"a": 1}\n</parameter>\n</function>\n</tool_call>'
        )
        parsed = parse_qwen_response(raw, tools=tools)
        assert parsed.tool_calls[0].arguments["content"] == '{"a": 1}'

    def test_content_around_call_preserved(self):
        raw = (
            "I will create the file.\n<tool_call>\n<function=write>\n"
            "<parameter=path>\na\n</parameter>\n</function>\n</tool_call>"
        )
        parsed = parse_qwen_response(raw)
        assert parsed.content == "I will create the file."
        assert parsed.tool_calls[0].name == "write"

    def test_thinking_then_xml_call_with_strip(self):
        raw = (
            "<think>\nplan it\n</think>\n<tool_call>\n<function=bash>\n"
            "<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"
        )
        parsed = parse_qwen_response(raw, strip_thinking=True)
        assert parsed.tool_calls[0].name == "bash"
        assert parsed.tool_calls[0].arguments == {"command": "ls -la"}
        assert parsed.content == ""


class TestParseFallback:
    def test_unparseable_body_stays_content(self):
        raw = "<tool_call>\nnot json and not xml\n</tool_call>"
        parsed = parse_qwen_response(raw)
        assert parsed.tool_calls == []
        assert parsed.finish_reason == "stop"
        assert "not json and not xml" in parsed.content


class TestParseTruncatedAndLooseCalls:
    def test_literal_newlines_in_json_string_args(self):
        raw = (
            '<tool_call>\n{"name": "write", "arguments": {"path": "app.py", '
            '"content": "line one\nline two\n    indented"}}\n</tool_call>'
        )
        parsed = parse_qwen_response(raw)
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].arguments["content"] == (
            "line one\nline two\n    indented"
        )

    def test_unterminated_json_call_recovered(self):
        raw = '<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la"}}'
        parsed = parse_qwen_response(raw)
        assert parsed.finish_reason == "tool_calls"
        assert parsed.tool_calls[0].name == "bash"
        assert parsed.tool_calls[0].arguments == {"command": "ls -la"}

    def test_unterminated_json_call_with_partial_close_tag(self):
        raw = '<tool_call>\n{"name": "bash", "arguments": {"command": "pwd"}}\n</tool_'
        parsed = parse_qwen_response(raw)
        assert parsed.tool_calls[0].arguments == {"command": "pwd"}

    def test_unterminated_xml_function_recovered(self):
        raw = (
            "<tool_call>\n<function=write>\n<parameter=path>\napp.py\n</parameter>\n"
            "<parameter=content>\nimport os\nprint(os.getcwd())"
        )
        parsed = parse_qwen_response(raw)
        assert parsed.finish_reason == "tool_calls"
        assert parsed.tool_calls[0].name == "write"
        assert parsed.tool_calls[0].arguments["path"] == "app.py"
        assert parsed.tool_calls[0].arguments["content"] == (
            "import os\nprint(os.getcwd())"
        )

    def test_unterminated_xml_with_closed_params_recovered(self):
        raw = (
            "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n"
            "</function>"
        )
        parsed = parse_qwen_response(raw)
        assert parsed.tool_calls[0].name == "bash"
        assert parsed.tool_calls[0].arguments == {"command": "ls"}

    def test_closed_call_after_truncation_fix_still_single(self):
        raw = (
            'before\n<tool_call>\n{"name": "ls", "arguments": {"path": "."}}\n'
            "</tool_call>\nafter"
        )
        parsed = parse_qwen_response(raw)
        assert len(parsed.tool_calls) == 1
        assert "before" in parsed.content
        assert "after" in parsed.content
