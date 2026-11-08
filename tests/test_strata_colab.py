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
import shutil
from contextlib import redirect_stdout
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
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


def configuration(**overrides):
    tree = source_tree()
    defaults = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {'QUANT', 'MAX_CONTEXT', 'KV_TYPE', 'KV_RESIDENT', 'PREFILL', 'ENABLE_MTP'}:
                defaults[name] = ast.literal_eval(node.value)
    start = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in node.targets))
    end = next(i for i, node in enumerate(tree.body)
               if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'config' for t in node.targets))
    ns = dict(PACK=Path('/pack'), GGUF=Path('/first.gguf'), PLE_GGUF=Path('/second.gguf'),
              ROOT=Path('/engine'), MTP=Path('/mtp'), VECTOR=Path('/vector.gguf'),
              STATE=Path('/state'), BUILD=Path('/build-sm89'), PORT=8080, **defaults)
    ns.update(overrides)
    exec(compile(ast.Module(body=tree.body[start:end + 1], type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns['config']


class NotebookTests(unittest.TestCase):
    def test_engine_version_check_rejects_a_stale_binary(self):
        ns = functions()
        with patch('subprocess.check_output', return_value='strata 0.1.58\n'):
            self.assertEqual(ns['check_engine_version'](Path('/build/strata'), '0.1.58'), 'strata 0.1.58')
        with patch('subprocess.check_output', return_value='strata 0.1.40.3\n'):
            with self.assertRaisesRegex(RuntimeError, '引擎版本不匹配'):
                ns['check_engine_version'](Path('/build/strata'), '0.1.58')

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
        config = configuration()
        args = config['args']
        self.assertEqual(config['model_name'], 'alesha-qwen3.8-flash-next-gsq-rco-iq3_xxs')
        self.assertEqual(args[args.index('--ple-gguf') + 1], '/second.gguf')
        self.assertEqual(args[args.index('--control-vector-scaled') + 1], '/vector.gguf:1.0')
        self.assertEqual(args[args.index('--control-vector-layer-range') + 1:args.index('--control-vector-layer-range') + 3], ['4', '44'])
        self.assertEqual(args[args.index('--cvec-mode') + 1], 'project')
        self.assertEqual(args[args.index('--prefill') + 1], 'auto')
        self.assertEqual(args[args.index('--spec') + 1], '4')
        self.assertEqual(args[args.index('--mtp') + 1], '/mtp/rt')
        self.assertEqual(args[args.index('--spec-min-p') + 1], '0.5')
        self.assertEqual(args[args.index('--lookup-chain') + 1], '4')
        self.assertEqual(args[args.index('--adapt-every') + 1], '1')
        self.assertEqual(args[args.index('--adapt-swaps') + 1], '96')
        self.assertIn('--prompt-cache-tail', args)
        self.assertEqual(args[args.index('--max-context') + 1], '262144')
        self.assertEqual(args[args.index('--kv-resident') + 1], '32768')
        self.assertEqual(args[args.index('--kv') + 1], 'k8v4')
        self.assertEqual(config['sampling'], {'experimental_speed_projection': True})
        self.assertEqual(config['drafting'], 'mtp')
        self.assertEqual(config['env']['STRATA_ROUTE_TAIL_SKIP'], '0')
        self.assertEqual(config['expert_profile_save'], '/state/expert-profile-learned.bin')
        self.assertEqual(config['expert_profile_save_every'], 5)

    def test_mtp_can_be_disabled_with_int8_kv_for_comparison(self):
        config = configuration(ENABLE_MTP=False, PREFILL=1024, KV_TYPE="int8")
        args = config['args']
        self.assertEqual(config['model_name'], 'alesha-qwen3.8-flash-next-gsq-rco-iq3_xxs')
        self.assertEqual(args[args.index('--prefill') + 1], '1024')
        self.assertEqual(args[args.index('--spec') + 1], '2')
        self.assertEqual(args[args.index('--kv') + 1], 'int8')
        self.assertNotIn('--mtp', args)
        self.assertNotIn('--spec-min-p', args)
        self.assertNotIn('--lookup-chain', args)
        self.assertEqual(config['drafting'], 'off')
        self.assertEqual(config['env']['STRATA_ROUTE_TAIL_SKIP'], '0')

    def test_incomplete_tokenizer_is_repaired_before_pack_is_marked_complete(self):
        tree = source_tree()
        start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'signature' for t in node.targets))
        end = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'MTP' for t in node.targets))
        block = compile(ast.Module(body=tree.body[start:end], type_ignores=[]), str(SOURCE), 'exec')
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            pack, model = data / 'pack', data / 'model'
            (pack / 'tokenizer').mkdir(parents=True)
            model.mkdir()
            (model / 'colab-download.json').write_text('{"revision": "model-revision", "files": []}')
            for name in ('vocab.json', 'chat_template.jinja'):
                (pack / 'tokenizer' / name).write_text('existing')
            calls = []

            def run_pack(args, env):
                calls.append(args[1].name)
                if args[1].name == 'iq_pack.py':
                    for name in ('index.txt', 'dense.bin', 'native_experts.txt', 'conversions.json'):
                        (pack / name).write_text('packed')
                elif args[1].name == 'strata_tokenizer.py':
                    for name in ('vocab.json', 'merges.txt', 'token_type.json', 'tokenizer.json', 'chat_template.jinja'):
                        (pack / 'tokenizer' / name).write_text('repaired')
                else:
                    self.fail(f'Unexpected tool: {args[1]}')

            ns = dict(STRATA_REV='release-revision', MODEL_DIR=model, PACK=pack, DATA=data,
                      ROOT=Path('/engine'), PY=Path('/python'), GGUF=Path('/model.gguf'),
                      ENV={}, Path=Path, json=json, shutil=shutil, run=run_pack)
            with patch('shutil.disk_usage', return_value=Mock(free=32 * 2**30)), redirect_stdout(io.StringIO()):
                exec(block, ns)
                exec(block, ns)
            self.assertEqual(calls, ['iq_pack.py', 'strata_tokenizer.py'])
            marker = json.loads((pack / 'colab-complete.json').read_text())
            self.assertEqual(marker['strata'], 'release-revision')

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
