from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .io import atomic_write_xml
from .models import base_function_for_nodeid, source_file_for_nodeid


@dataclass(frozen=True)
class TestCaseResult:
    nodeid: str
    source_file: str
    base_function: str
    outcome: str
    seconds: float
    synthetic: bool
    diagnostic: str


@dataclass(frozen=True)
class BatchValidation:
    valid: bool
    cases: tuple[TestCaseResult, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class SyntheticBatchMetadata:
    policy: str
    batch_id: str
    unit_id: str
    shard_index: int
    attempt_id: str
    profile: str


def test_suites(root: ET.Element) -> list[ET.Element]:
    if root.tag == "testsuite":
        return [root]
    if root.tag == "testsuites":
        return list(root.findall("./testsuite"))
    return []


def element_properties(element: ET.Element) -> dict[str, str]:
    return {
        prop.attrib.get("name", ""): prop.attrib.get("value", "")
        for prop in element.findall("./properties/property")
    }


def testcase_outcome(testcase: ET.Element) -> str:
    if testcase.find("./failure") is not None or testcase.find("./error") is not None:
        return "failed"
    if testcase.find("./skipped") is not None:
        return "skipped"
    return "passed"


def validate_batch_xml(path: Path, expected_nodeids: Sequence[str]) -> BatchValidation:
    diagnostics: list[str] = []
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        return BatchValidation(False, (), (f"malformed JUnit XML: {error}",))
    suites = test_suites(root)
    if not suites:
        return BatchValidation(False, (), (f"unexpected JUnit root {root.tag!r}",))
    cases: list[TestCaseResult] = []
    seen: list[str] = []
    for suite in suites:
        for testcase in suite.findall("./testcase"):
            properties = element_properties(testcase)
            nodeid = properties.get("pytest_nodeid")
            if not nodeid:
                diagnostics.append("testcase is missing pytest_nodeid property")
                continue
            seen.append(nodeid)
            try:
                seconds = float(testcase.attrib.get("time", "0") or 0)
            except ValueError:
                diagnostics.append(f"testcase {nodeid} has invalid time")
                seconds = 0.0
            cases.append(
                TestCaseResult(
                    nodeid=nodeid,
                    source_file=source_file_for_nodeid(nodeid),
                    base_function=base_function_for_nodeid(nodeid),
                    outcome=testcase_outcome(testcase),
                    seconds=seconds,
                    synthetic=properties.get("synthetic") == "true",
                    diagnostic=_testcase_diagnostic(testcase),
                )
            )
    expected = set(expected_nodeids)
    counts = Counter(seen)
    actual = set(counts)
    duplicates = sorted(nodeid for nodeid, count in counts.items() if count > 1)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        diagnostics.append("missing planned nodes: " + ", ".join(missing[:5]))
    if unexpected:
        diagnostics.append("unexpected nodes: " + ", ".join(unexpected[:5]))
    if duplicates:
        diagnostics.append("duplicate nodes: " + ", ".join(duplicates[:5]))
    return BatchValidation(not diagnostics, tuple(cases), tuple(diagnostics))


def _testcase_diagnostic(testcase: ET.Element) -> str:
    result = testcase.find("./failure")
    if result is None:
        result = testcase.find("./error")
    if result is None:
        return ""
    message = result.attrib.get("message", "").strip()
    detail = (result.text or "").strip()
    if message and detail and message not in detail:
        return f"{message}\n{detail}"
    return detail or message


def finalized_batch_outcomes(
    path: Path, expected_nodeids: Sequence[str]
) -> tuple[Counter[str], tuple[str, ...]]:
    validation = validate_batch_xml(path, expected_nodeids)
    if not validation.valid:
        return Counter(), validation.diagnostics

    recorded: dict[str, str] = {}
    result_path = path.with_name(f"{path.stem}.results.json")
    try:
        value = json.loads(result_path.read_text(encoding="utf-8"))
        results = value.get("results", [])
        if isinstance(results, list):
            for result in results:
                if not isinstance(result, dict):
                    continue
                nodeid = result.get("nodeid")
                outcome = result.get("outcome")
                if isinstance(nodeid, str) and isinstance(outcome, str):
                    recorded[nodeid] = (
                        outcome
                        if outcome in {"passed", "failed", "skipped", "unknown"}
                        else "unknown"
                    )
    except (OSError, AttributeError, json.JSONDecodeError):
        pass

    return (
        Counter(recorded.get(case.nodeid, case.outcome) for case in validation.cases),
        (),
    )


def _add_properties(element: ET.Element, values: dict[str, str]) -> None:
    properties = element.find("./properties")
    if properties is None:
        properties = ET.Element("properties")
        element.insert(0, properties)
    existing = element_properties(element)
    for name, value in values.items():
        if name not in existing:
            ET.SubElement(properties, "property", name=name, value=value)


def annotate_batch_xml(path: Path, values: dict[str, str]) -> None:
    root = ET.parse(path).getroot()
    for suite in test_suites(root):
        _add_properties(suite, values)
    if root.tag == "testsuite":
        wrapper = ET.Element("testsuites", root.attrib)
        wrapper.append(root)
        root = wrapper
    atomic_write_xml(path, root)


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# XML 1.0 allows tab, newline, carriage return, and everything from 0x20 up.
_XML_ILLEGAL = re.compile(r"[^\x09\x0a\x0d\x20-\U0010ffff]")


def _xml_safe_text(value: str) -> str:
    """Strip terminal colouring and codepoints XML 1.0 cannot represent.

    pytest renders ``longrepr`` with ANSI colour, which is legal XML but noise,
    and tracebacks can carry control characters that make the document
    unparseable. A report we cannot parse is worse than no report, because the
    failure it records is then invisible.
    """

    return _XML_ILLEGAL.sub("", _ANSI_ESCAPE.sub("", value))


def create_collection_error_xml(
    path: Path,
    errors: Sequence[dict[str, str]],
    skips: Sequence[dict[str, str]] = (),
) -> None:
    """Write one testcase per module that produced no collected items.

    A module contributes zero nodes either because it raised while importing
    (reported here as a failure) or because it skipped itself at module level
    via ``pytest.importorskip`` / ``pytest.skip(allow_module_level=True)``
    (reported as a skip, which is what a plain pytest run would show). Without
    this the module is absent from the report entirely, which reads as "these
    tests passed" rather than "these tests never ran".
    """

    total = len(errors) + len(skips)
    attributes = {
        "tests": str(total),
        "failures": str(len(errors)),
        "errors": "0",
        "skipped": str(len(skips)),
        "time": "0",
    }
    suite = ET.Element("testsuite", name="collection-errors", **attributes)
    _add_properties(suite, {"synthetic": "true", "collection_error": "true"})

    def _case(entry: dict[str, str], outcome: str) -> None:
        source_file = entry.get("source_file", "<unknown>")
        message = entry.get("message", f"collection {outcome}")
        testcase = ET.SubElement(
            suite,
            "testcase",
            classname=source_file.replace("/", "."),
            name="<collection-error>" if outcome == "failure" else "<collection-skip>",
            time="0",
        )
        _add_properties(
            testcase,
            {
                "pytest_nodeid": f"{source_file}::{testcase.get('name')}",
                "synthetic": "true",
                "collection_error": "true",
            },
        )
        summary_text = (
            f"{source_file} could not be collected"
            if outcome == "failure"
            else f"{source_file} skipped itself at module level"
        )
        result = ET.SubElement(testcase, outcome, message=summary_text)
        result.text = _xml_safe_text(message)

    for entry in errors:
        _case(entry, "failure")
    for entry in skips:
        _case(entry, "skipped")
    root = ET.Element("testsuites", **attributes)
    root.append(suite)
    atomic_write_xml(path, root)


def create_synthetic_batch_xml(
    path: Path,
    nodeids: Sequence[str],
    metadata: SyntheticBatchMetadata,
) -> None:
    policy = metadata.policy
    if policy not in {"skip", "fail"}:
        raise ValueError("synthetic policy must be skip or fail")
    suite = ET.Element(
        "testsuite",
        name=metadata.batch_id,
        tests=str(len(nodeids)),
        failures=str(len(nodeids) if policy == "fail" else 0),
        errors="0",
        skipped=str(len(nodeids) if policy == "skip" else 0),
        time="0",
    )
    _add_properties(
        suite,
        {
            "batch_id": metadata.batch_id,
            "unit_id": metadata.unit_id,
            "shard_index": str(metadata.shard_index),
            "attempt_id": metadata.attempt_id,
            "timing_profile": metadata.profile,
            "synthetic": "true",
        },
    )
    for nodeid in nodeids:
        testcase = ET.SubElement(
            suite,
            "testcase",
            classname=source_file_for_nodeid(nodeid).replace("/", "."),
            name=nodeid.split("::")[-1],
            time="0",
        )
        _add_properties(
            testcase,
            {
                "pytest_nodeid": nodeid,
                "synthetic": "true",
                "timeout_policy": policy,
            },
        )
        result_tag = "skipped" if policy == "skip" else "failure"
        result = ET.SubElement(
            testcase,
            result_tag,
            message="not executed due to timeout",
        )
        result.text = "not executed due to timeout"
    root = ET.Element(
        "testsuites",
        tests=str(len(nodeids)),
        failures=str(len(nodeids) if policy == "fail" else 0),
        errors="0",
        skipped=str(len(nodeids) if policy == "skip" else 0),
        time="0",
    )
    root.append(suite)
    atomic_write_xml(path, root)
