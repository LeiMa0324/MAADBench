import json

def _parse_response(self, response: str):
    # todo: many failure caused by parsing failure
    def _extract_json_text(resp: str) -> str:
        if "```json" in resp:
            return resp.split("```json", 1)[1].split("```", 1)[0].strip()
        if "```" in resp:
            return resp.split("```", 1)[1].split("```", 1)[0].strip()
        return resp.strip()

    def _coerce_to_obj(s: str):
        """
        Try to coerce s into a Python object (dict/list/primitive),
        handling:
        1) normal JSON
        2) double-encoded JSON string
        3) escaped JSON without outer quotes: {\\\"a\\\":1}
        """
        s = s.strip()

        # Try 1: normal JSON
        try:
            obj = json.loads(s)
        except Exception:
            obj = None

        # Try 2: double-encoded JSON string
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except Exception:
                pass

        if obj is not None:
            return obj

        # Try 3: escaped-json-without-quotes, e.g. {\"a\":1}
        # Heuristic: contains \", and starts with {\" or [\"
        if '\\"' in s and (
                s.startswith('{\\\"') or s.startswith('[\\\"') or s.startswith('\\{\"') or s.startswith('\\[')):
            # Some models prepend a backslash before { ... } ; normalize
            if s.startswith('\\'):
                s = s[1:]

            # Wrap as a JSON string literal to unescape safely
            try:
                unescaped = json.loads(f"\"{s}\"")  # turns {\"a\":1} -> {"a":1}
                obj2 = json.loads(unescaped)
                return obj2
            except Exception:
                pass

        # Last resort: sometimes it's almost JSON but with leading/trailing junk
        # (you can add regex extraction later if needed)
        raise ValueError("Unable to coerce model output into JSON")

    try:
        json_str = _extract_json_text(response)
        parsed = _coerce_to_obj(json_str)

        if isinstance(parsed, dict):
            confidence = parsed.pop("confidence", 0.7)
            content = parsed
        else:
            confidence = 0.7
            content = {"result": parsed}

        content = content


    except Exception as e:
        content = {"error": "Failed to parse", "raw": response, "exception": str(e)}

    return content

