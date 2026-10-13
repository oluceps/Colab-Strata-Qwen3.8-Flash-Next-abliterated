import io
import unittest
from unittest.mock import patch, MagicMock
from urllib.parse import unquote

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import open_colab_browser as browser


class BrowserTests(unittest.TestCase):
    def test_fragment_reaches_cdp_without_a_proxy(self):
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value = io.StringIO('{"id":"tab"}')
        url = "https://colab.research.google.com/notebooks/empty.ipynb#mcpProxyToken=test&mcpProxyPort=12345"
        with patch.object(browser, "build_opener", return_value=opener) as build:
            browser.open_colab(url)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), "PUT")
        self.assertEqual(unquote(request.full_url.split("?", 1)[1]), url)
        self.assertEqual(build.call_args.args[0].proxies, {})

    def test_other_origins_are_rejected(self):
        with patch.object(browser, "build_opener") as build:
            with self.assertRaises(ValueError):
                browser.open_colab("https://example.com/")
            build.assert_not_called()

    def test_error_does_not_print_connection_token(self):
        output = io.StringIO()
        with patch.object(browser.sys, "argv", ["open_colab_browser.py", "private-token"]), \
             patch.object(browser, "open_colab", side_effect=RuntimeError("private-token")), \
             patch.object(browser.sys, "stderr", output):
            self.assertEqual(browser.main(), 1)
        self.assertNotIn("private-token", output.getvalue())


if __name__ == "__main__":
    unittest.main()
