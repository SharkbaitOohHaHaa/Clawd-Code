from __future__ import annotations

import unittest

from src.agent.session import Session


class TestSessionPathSecurity(unittest.TestCase):
    def test_agent_session_load_rejects_path_traversal(self) -> None:
        with self.assertRaises(ValueError):
            Session.load(r"..\config")


if __name__ == "__main__":
    unittest.main()
