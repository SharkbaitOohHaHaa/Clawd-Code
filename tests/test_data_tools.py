from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.errors import ToolInputError
from src.tool_system.protocol import ToolCall
from src.tool_system.tools import DataInspectTool, DataTransformTool


class DataToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_tools_are_registered_with_checked_contracts(self) -> None:
        registry = build_default_registry(include_user_tools=False)
        inspect_spec = registry.get("DataInspect").spec()
        transform_spec = registry.get("DataTransform").spec()

        self.assertEqual(len(registry.list_specs()), 44)
        self.assertEqual(inspect_spec.permission_policy, "checked")
        self.assertTrue(inspect_spec.is_read_only)
        self.assertEqual(transform_spec.permission_policy, "checked")
        self.assertTrue(transform_spec.is_destructive)

    def test_inspect_csv_reports_schema_preview_and_header_only_columns(self) -> None:
        csv_path = self.root / "people.csv"
        csv_path.write_text(
            "name,age\nalice,30\nbob,40\n",
            encoding="utf-8",
        )
        result = DataInspectTool().run(
            {"file_path": str(csv_path), "preview_rows": 1},
            self.ctx,
        )
        output = result.output

        self.assertFalse(result.is_error)
        self.assertEqual(output["format"], "csv")
        self.assertEqual(output["rowCount"], 2)
        self.assertEqual(output["previewRows"], 1)
        self.assertTrue(output["previewTruncated"])
        self.assertEqual(
            output["rows"],
            [{"name": "alice", "age": "30"}],
        )
        self.assertEqual(
            output["columns"],
            [
                {"name": "name", "types": ["string"], "nullable": False},
                {"name": "age", "types": ["string"], "nullable": False},
            ],
        )

        header_only = self.root / "empty.csv"
        header_only.write_text("name,age\n", encoding="utf-8")
        header_result = DataInspectTool().run(
            {"file_path": str(header_only)},
            self.ctx,
        ).output
        self.assertEqual(header_result["rowCount"], 0)
        self.assertEqual(
            [item["name"] for item in header_result["columns"]],
            ["name", "age"],
        )

    def test_inspect_tsv_json_and_jsonl(self) -> None:
        tsv = self.root / "data.tsv"
        tsv.write_text("id\tvalue\n1\ta\n", encoding="utf-8")
        json_path = self.root / "data.json"
        json_path.write_text(
            json.dumps(
                [
                    {"id": 1, "active": True, "note": None},
                    {"id": 2, "active": False, "note": "ok"},
                ]
            ),
            encoding="utf-8",
        )
        jsonl = self.root / "data.jsonl"
        jsonl.write_text(
            '{"id":1,"kind":"a"}\n{"id":2,"kind":"b"}\n',
            encoding="utf-8",
        )

        tsv_out = DataInspectTool().run({"file_path": str(tsv)}, self.ctx).output
        json_out = DataInspectTool().run(
            {"file_path": str(json_path)},
            self.ctx,
        ).output
        jsonl_out = DataInspectTool().run(
            {"file_path": str(jsonl)},
            self.ctx,
        ).output

        self.assertEqual(tsv_out["format"], "tsv")
        self.assertEqual(tsv_out["rows"], [{"id": "1", "value": "a"}])
        self.assertEqual(jsonl_out["format"], "jsonl")
        self.assertEqual(jsonl_out["rowCount"], 2)
        self.assertEqual(
            json_out["columns"],
            [
                {"name": "id", "types": ["integer"], "nullable": False},
                {"name": "active", "types": ["boolean"], "nullable": False},
                {
                    "name": "note",
                    "types": ["null", "string"],
                    "nullable": True,
                },
            ],
        )

    def test_transform_csv_filters_projects_and_writes_new_jsonl(self) -> None:
        source = self.root / "people.csv"
        source.write_text(
            "name,age,city\nalice,30,Boston\nbob,40,Lowell\n",
            encoding="utf-8",
        )
        target = self.root / "selected.jsonl"
        DataInspectTool().run({"file_path": str(source)}, self.ctx)

        result = DataTransformTool().run(
            {
                "input_path": str(source),
                "output_path": str(target),
                "columns": ["name", "city"],
                "filters": {"age": "40"},
            },
            self.ctx,
        )

        self.assertFalse(result.is_error)
        self.assertEqual(result.output["inputRows"], 2)
        self.assertEqual(result.output["outputRows"], 1)
        self.assertEqual(result.output["columns"], ["name", "city"])
        self.assertEqual(result.output["filterColumns"], ["age"])
        self.assertEqual(
            json.loads(target.read_text(encoding="utf-8").strip()),
            {"name": "bob", "city": "Lowell"},
        )

    def test_transform_json_uses_typed_exact_filter_and_writes_tsv(self) -> None:
        source = self.root / "rows.json"
        source.write_text(
            json.dumps(
                [
                    {"id": 1, "active": True, "score": 1.5},
                    {"id": 2, "active": False, "score": 2.5},
                ]
            ),
            encoding="utf-8",
        )
        target = self.root / "active.tsv"
        DataInspectTool().run({"file_path": str(source)}, self.ctx)

        DataTransformTool().run(
            {
                "input_path": str(source),
                "output_path": str(target),
                "columns": ["id", "score"],
                "filters": {"active": True},
            },
            self.ctx,
        )

        self.assertEqual(target.read_text(encoding="utf-8"), "id\tscore\n1\t1.5\n")

    def test_transform_requires_read_unchanged_source(self) -> None:
        source = self.root / "source.csv"
        source.write_text("id,value\n1,a\n", encoding="utf-8")
        target = self.root / "out.json"
        tool = DataTransformTool()
        payload = {
            "input_path": str(source),
            "output_path": str(target),
        }

        with self.assertRaises(ToolInputError):
            tool.run(payload, self.ctx)

        DataInspectTool().run({"file_path": str(source)}, self.ctx)
        source.write_text("id,value\n1,b\n", encoding="utf-8")
        with self.assertRaises(ToolInputError):
            tool.run(payload, self.ctx)

    def test_transform_refuses_existing_same_or_invalid_targets(self) -> None:
        source = self.root / "source.csv"
        source.write_text("id,value\n1,a\n", encoding="utf-8")
        DataInspectTool().run({"file_path": str(source)}, self.ctx)
        existing = self.root / "existing.json"
        existing.write_text("[]\n", encoding="utf-8")
        tool = DataTransformTool()

        with self.assertRaises(ToolInputError):
            tool.run(
                {
                    "input_path": str(source),
                    "output_path": str(existing),
                },
                self.ctx,
            )
        with self.assertRaises(ToolInputError):
            tool.run(
                {
                    "input_path": str(source),
                    "output_path": str(source),
                },
                self.ctx,
            )
        with self.assertRaises(ToolInputError):
            tool.run(
                {
                    "input_path": str(source),
                    "output_path": str(self.root / "unsupported.parquet"),
                },
                self.ctx,
            )

    def test_transform_empty_json_can_seed_delimited_header(self) -> None:
        source = self.root / "empty.json"
        source.write_text("[]\n", encoding="utf-8")
        target = self.root / "empty.csv"
        DataInspectTool().run({"file_path": str(source)}, self.ctx)

        result = DataTransformTool().run(
            {
                "input_path": str(source),
                "output_path": str(target),
                "columns": ["id", "value"],
            },
            self.ctx,
        )

        self.assertEqual(result.output["outputRows"], 0)
        self.assertEqual(target.read_text(encoding="utf-8"), "id,value\n")

    def test_rejects_malformed_data_unknown_columns_and_non_scalar_filters(self) -> None:
        duplicate = self.root / "duplicate.csv"
        duplicate.write_text("id,id\n1,2\n", encoding="utf-8")
        with self.assertRaises(ToolInputError):
            DataInspectTool().run({"file_path": str(duplicate)}, self.ctx)

        bad_jsonl = self.root / "bad.jsonl"
        bad_jsonl.write_text('{"id":1}\nnot-json\n', encoding="utf-8")
        with self.assertRaises(ToolInputError):
            DataInspectTool().run({"file_path": str(bad_jsonl)}, self.ctx)

        source = self.root / "source.json"
        source.write_text('[{"id":1,"kind":"a"}]\n', encoding="utf-8")
        DataInspectTool().run({"file_path": str(source)}, self.ctx)
        tool = DataTransformTool()

        with self.assertRaises(ToolInputError):
            tool.run(
                {
                    "input_path": str(source),
                    "output_path": str(self.root / "unknown.json"),
                    "columns": ["missing"],
                },
                self.ctx,
            )
        with self.assertRaises(ToolInputError):
            tool.run(
                {
                    "input_path": str(source),
                    "output_path": str(self.root / "nested.json"),
                    "filters": {"kind": {"nested": True}},
                },
                self.ctx,
            )

    def test_registry_dispatch_inspect_then_transform(self) -> None:
        source = self.root / "source.jsonl"
        source.write_text('{"id":1}\n{"id":2}\n', encoding="utf-8")
        target = self.root / "result.json"
        registry = build_default_registry(include_user_tools=False)

        inspected = registry.dispatch(
            ToolCall(name="DataInspect", input={"file_path": str(source)}),
            self.ctx,
        )
        transformed = registry.dispatch(
            ToolCall(
                name="DataTransform",
                input={
                    "input_path": str(source),
                    "output_path": str(target),
                    "filters": {"id": 2},
                },
            ),
            self.ctx,
        )

        self.assertFalse(inspected.is_error)
        self.assertFalse(transformed.is_error)
        self.assertEqual(
            json.loads(target.read_text(encoding="utf-8")),
            [{"id": 2}],
        )


if __name__ == "__main__":
    unittest.main()
