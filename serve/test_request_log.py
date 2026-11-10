import json
import tempfile
import unittest
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve


class RequestLogFiles(unittest.TestCase):
    """"request_log_dir": a file per request, in the vLLM logs' format, with the monitor off."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "Thought.</think>\nHello there.", max_context=16384), tok,
                           ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.request_log_dir = self.dir.name
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def post(self, body):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer sekrit"})
        return urllib.request.urlopen(req, timeout=10).read()

    def one_log(self):
        files = list(Path(self.dir.name).iterdir())
        self.assertEqual(len(files), 1)
        return files[0].name, files[0].read_text()

    def check(self, stream):
        body = {"messages": [{"role": "system", "content": "You are Hermes Agent, built by tests"},
                             {"role": "user", "content": "hi"}], "max_tokens": 64, "stream": stream}
        self.post(body)
        name, text = self.one_log()
        self.assertRegex(name, r"^\d{4}-\d\d-\d\dT\d\d-\d\d-\d\dZ_chat\.plain\.[0-9a-f]{12}_SYSTEM_You_are_Hermes_Agent")
        self.assertNotIn("sekrit", text)
        for section in ("=== HEADERS ===", "=== REQUEST ===", "=== RESPONSE (streaming) ===", "=== PERFORMANCE ==="):
            self.assertIn(section, text)
        self.assertIn("Hello there.", text)
        self.assertIn("<think>", text)
        self.assertRegex(text, r"Prompt tokens +: \d+")
        self.assertRegex(text, r"Time to first token +: [\d.]+ s")
        self.assertRegex(text, r"Outcome +: completed")
        # the replay's reader takes the body back
        rest = text.partition("=== REQUEST ===")[2]
        self.assertEqual(json.JSONDecoder().raw_decode(rest.lstrip())[0], body)

    def test_stream(self):
        self.check(True)

    def test_non_stream(self):
        self.check(False)

    def test_tool_calls_land_whole_in_their_own_section(self):
        """#4: a streamed tool call arrives in token-sized pieces: the log must show the whole calls under
        === TOOL CALLS ===, not a [tool_call: ...] wrapper around every piece inside the RESPONSE section."""
        tok = ByteTokenizer()
        call = ("Thought.</think>\n\n<tool_call>\n<function=search>\n<parameter=q>\n2+2\n"
                "</parameter>\n</function>\n</tool_call>")
        self.svc.engine = MockEngine(tok, call, max_context=16384)
        for stream in (True, False):
            with self.subTest(stream=stream):
                for old in Path(self.dir.name).iterdir():       # one_log() wants exactly this request's file
                    old.unlink()
                body = {"messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 256, "stream": stream,
                        "tools": [{"type": "function", "function": {"name": "search", "description": "search",
                                                                    "parameters": {"type": "object",
                                                                                   "properties": {"q": {"type":
                                                                                            "string"}}}}}]}
                raw = self.post(body).decode()
                self.assertIn('"finish_reason":"tool_calls"', raw.replace(" ", ""))
                _, text = self.one_log()
                self.assertIn("=== TOOL CALLS ===", text)
                self.assertNotIn("[tool_call", text)
                section = text.partition("=== TOOL CALLS ===")[2].partition("\n\n")[0]
                calls = json.loads(section)
                self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls],
                                 [("search", {"q": "2+2"})])
                self.assertTrue(all(c["id"] for c in calls))
                self.assertRegex(text, r"Finish reason +: tool_calls")
                # the thinking still rides in the RESPONSE section, the call pieces no longer break it up
                self.assertIn("Thought.", text)


if __name__ == "__main__":
    unittest.main()
