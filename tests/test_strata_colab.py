import ast
import json
import os
from pathlib import Path
import signal
import socket
import errno
import subprocess
import sys
import io
from contextlib import redirect_stdout
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.request


SOURCE = Path(__file__).resolve().parents[1] / "strata_l4_gsq_rco_iq3_xxs.py"


def source_tree():
    source = "\n".join(line for line in SOURCE.read_text().splitlines() if not line.startswith("!"))
    return ast.parse(source)


def functions():
    tree = source_tree()
    definitions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    namespace = dict(Path=Path, os=os, json=json, signal=signal, time=time,
                     subprocess=subprocess, socket=socket, errno=errno, urllib=__import__("urllib"))
    exec(compile(definitions, str(SOURCE), "exec"), namespace)
    return namespace


class NotebookTests(unittest.TestCase):
    def test_occupied_port_selects_another_without_stopping_listener(self):
        ns = functions()
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            occupied = listener.getsockname()[1]
            chosen = ns['select_available_port'](occupied)
            self.assertNotEqual(chosen, occupied)
            with socket.create_connection(('127.0.0.1', occupied), timeout=1):
                pass

    def test_gsq_config_uses_second_shard_and_projection(self):
        tree = source_tree()
        start = next(i for i, node in enumerate(tree.body)
                     if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets))
        end = next(i for i, node in enumerate(tree.body)
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "config" for t in node.targets))
        ns = dict(PACK=Path('/pack'), GGUF=Path('/first.gguf'), PLE_GGUF=Path('/second.gguf'),
                  ROOT=Path('/engine'), PREFILL=512, ENABLE_MTP=False, MAX_CONTEXT=131072,
                  KV_RESIDENT=32768,
                  VECTOR=Path('/vector.gguf'), QUANT='IQ3_S', STATE=Path('/state'),
                  BUILD=Path('/build-sm89'), PORT=8080)
        exec(compile(ast.Module(body=tree.body[start:end + 1], type_ignores=[]), str(SOURCE), 'exec'), ns)
        args = ns['config']['args']
        self.assertEqual(args[args.index('--ple-gguf') + 1], '/second.gguf')
        self.assertEqual(args[args.index('--control-vector-scaled') + 1], '/vector.gguf:1.0')
        self.assertEqual(args[args.index('--control-vector-layer-range') + 1:args.index('--control-vector-layer-range') + 3], ['4', '44'])
        self.assertEqual(args[args.index('--cvec-mode') + 1], 'project')
        self.assertEqual(args[args.index('--spec') + 1], '2')
        self.assertNotIn('--mtp', args)
        self.assertEqual(args[args.index('--max-context') + 1], '131072')
        self.assertEqual(args[args.index('--kv-resident') + 1], '32768')
        self.assertTrue(ns['config']['sampling']['experimental_speed_projection'])

    def test_embedded_download_script_compiles(self):
        tree = source_tree()
        node = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'download_script' for t in n.targets))
        compile(ast.literal_eval(node.value), '<download>', 'exec')

    def test_run_shows_child_stdout_and_stderr(self):
        ns = functions()
        output = io.StringIO()
        with redirect_stdout(output):
            ns["run"]([sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"])
        self.assertIn("out", output.getvalue())
        self.assertIn("err", output.getvalue())

    def test_run_failure_shows_cause_without_dumping_command(self):
        ns = functions()
        output = io.StringIO()
        code = "import sys; print('download failed', file=sys.stderr); sys.exit(7)"
        with redirect_stdout(output), self.assertRaises(RuntimeError) as error:
            ns["run"]([sys.executable, "-c", code])
        self.assertIn("download failed", output.getvalue())
        self.assertIn("7", str(error.exception))
        self.assertNotIn(code, str(error.exception))

    def test_cells_compile(self):
        notebook = json.loads(SOURCE.with_suffix(".ipynb").read_text())
        code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code), 6)
        for cell in code:
            if not cell.startswith("!"):
                ast.parse(cell)
        for cell in notebook["cells"]:
            self.assertFalse(cell.get("outputs"))

    def test_process_identity_matches_linux_starttime(self):
        ns = functions()
        expected = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
        self.assertEqual(ns["process_identity"](os.getpid()), expected)
        self.assertIsNone(ns["process_identity"](999999999))

    def test_reused_pid_is_not_killed(self):
        ns = functions()
        with tempfile.TemporaryDirectory() as d:
            ns["STATE"] = Path(d)
            record = Path(d) / "server.pid.json"
            record.write_text(json.dumps({"pid": 12345, "start": "old"}))
            ns["process_identity"] = lambda pid: "new"
            with patch("os.killpg") as kill:
                ns["stop_owned"]("server")
                kill.assert_not_called()
            self.assertFalse(record.exists())

    def test_owned_process_receives_term(self):
        ns = functions()
        with tempfile.TemporaryDirectory() as d:
            ns["STATE"] = Path(d)
            record = Path(d) / "server.pid.json"
            record.write_text(json.dumps({"pid": 12345, "start": "owned"}))
            identities = iter(["owned", None])
            ns["process_identity"] = lambda pid: next(identities)
            with patch("os.killpg") as kill:
                ns["stop_owned"]("server")
                kill.assert_called_once_with(12345, signal.SIGTERM)
            self.assertFalse(record.exists())

    def test_request_auth_is_optional_only_when_explicit(self):
        ns = functions()
        ns.update(API_KEY="test-key", PORT=8080)
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = b'{"status":"ok"}'
            self.assertEqual(ns["request"]("/health"), {"status": "ok"})
            request = urlopen.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
            ns["request"]("/v1/models", authenticated=False)
            self.assertIsNone(urlopen.call_args.args[0].get_header("Authorization"))


if __name__ == "__main__":
    unittest.main()
