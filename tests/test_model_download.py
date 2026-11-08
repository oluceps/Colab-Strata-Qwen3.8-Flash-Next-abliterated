import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "strata_l4_gsq_rco_iq3_xxs.py"
NAMES = ["IQ3_XXS/first.gguf", "IQ3_XXS/second.gguf", "vector.gguf"]
PAYLOAD = b"GGUFtest"


def run_download(destination, download, free=100 * 2**30):
    tree = ast.parse(SOURCE.read_text().split("\n!cat ", 1)[0])
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'download_script' for t in n.targets))
    script = ast.literal_eval(node.value)
    info = SimpleNamespace(sha='pinned-revision', siblings=[
        SimpleNamespace(rfilename=name, size=len(PAYLOAD)) for name in NAMES])
    api = SimpleNamespace(model_info=Mock(return_value=info))
    module = ModuleType('huggingface_hub')
    module.HfApi = Mock(return_value=api)
    module.hf_hub_download = Mock(side_effect=download)
    with patch.dict(sys.modules, huggingface_hub=module), \
            patch.object(sys, 'argv', ['download', 'test/model', str(destination), json.dumps(NAMES), '2']), \
            patch('shutil.disk_usage', return_value=SimpleNamespace(free=free)), \
            redirect_stdout(io.StringIO()):
        exec(compile(script, '<download>', 'exec'), {})
    return module


def write_download(**kwargs):
    target = Path(kwargs['local_dir']) / kwargs['filename']
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(PAYLOAD)
    return str(target)


class DownloadTests(unittest.TestCase):
    def test_shards_run_concurrently_and_existing_vector_is_reused(self):
        barrier = threading.Barrier(2)

        def download(**kwargs):
            barrier.wait(timeout=3)
            return write_download(**kwargs)

        with tempfile.TemporaryDirectory() as d:
            destination = Path(d)
            (destination / NAMES[2]).write_bytes(PAYLOAD)
            module = run_download(destination, download)
            calls = module.hf_hub_download.call_args_list
            self.assertEqual({c.kwargs['filename'] for c in calls}, set(NAMES[:2]))
            self.assertTrue(all(c.kwargs['revision'] == 'pinned-revision' for c in calls))
            self.assertTrue(all(c.kwargs['local_dir'] == destination for c in calls))
            manifest = json.loads((destination / 'colab-download.json').read_text())
            self.assertEqual(manifest['revision'], 'pinned-revision')
            self.assertEqual([f['name'] for f in manifest['files']], NAMES)
            self.assertFalse((destination / 'colab-download.tmp').exists())

    def test_complete_files_do_not_download_again(self):
        with tempfile.TemporaryDirectory() as d:
            destination = Path(d)
            for name in NAMES:
                write_download(local_dir=destination, filename=name)
            module = run_download(destination, write_download)
            module.hf_hub_download.assert_not_called()

    def test_disk_budget_covers_all_parallel_downloads_before_start(self):
        download = Mock(side_effect=write_download)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, '可用磁盘'):
                run_download(Path(d), download, free=24 * 2**30 + 2 * len(PAYLOAD))
            download.assert_not_called()
            self.assertFalse((Path(d) / 'colab-download.json').exists())

    def test_invalid_gguf_does_not_write_completion_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            destination = Path(d)
            target = destination / NAMES[0]
            target.parent.mkdir(parents=True)
            target.write_bytes(b'NOPEtest')
            with self.assertRaisesRegex(RuntimeError, '不是 GGUF'):
                run_download(destination, write_download)
            self.assertFalse((destination / 'colab-download.json').exists())

    def test_incomplete_transfer_does_not_write_completion_manifest(self):
        def download(**kwargs):
            result = write_download(**kwargs)
            if kwargs['filename'] == NAMES[0]:
                Path(result).write_bytes(b'GGUF')
            return result

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, '大小不符'):
                run_download(Path(d), download)
            self.assertFalse((Path(d) / 'colab-download.json').exists())


if __name__ == '__main__':
    unittest.main()
