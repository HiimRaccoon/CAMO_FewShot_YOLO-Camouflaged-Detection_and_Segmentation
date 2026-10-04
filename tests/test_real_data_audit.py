import os
from pathlib import Path
import hashlib
from copy import deepcopy

import pytest

from camo_fs.annotations import Taxonomy, audit_shot, audit_test
from camo_fs.paths import DatasetPaths
from camo_fs.segments import annotation_to_yolo


pytestmark = pytest.mark.real_data


@pytest.mark.real_data
def test_official_splits_match_audited_counts(tmp_path: Path) -> None:
    data_root_value = os.environ.get("CAMO_FS_DATA_ROOT")
    if data_root_value is None:
        pytest.skip("Set CAMO_FS_DATA_ROOT to run the read-only official-data audit")

    paths = DatasetPaths.from_root(Path(data_root_value), tmp_path / "work")
    taxonomy = Taxonomy.from_test_json(paths.test_json)
    assert len(taxonomy.names) == 47

    expected_counts = {1: (47, 47), 2: (94, 94), 3: (141, 141), 5: (197, 235)}
    for shot, (image_count, annotation_count) in expected_counts.items():
        report = audit_shot(shot, paths, taxonomy)
        assert not report.errors
        assert (report.image_count, report.annotation_count) == (image_count, annotation_count)
        assert len(report.source_files) == 47
        if shot == 5:
            assert any(issue.code == "reused_annotation_id" for issue in report.warnings)


@pytest.mark.parametrize("shot", [0, 1, 2, 3, 5], ids=["test", "1shot", "2shot", "3shot", "5shot"])
def test_official_geometry_conversion_preflight_is_read_only(tmp_path, shot):
    data_root_value = os.environ.get("CAMO_FS_DATA_ROOT")
    if data_root_value is None:
        pytest.skip("Set CAMO_FS_DATA_ROOT to run official test/train geometry conversion")
    paths = DatasetPaths.from_root(Path(data_root_value), tmp_path / "work")
    taxonomy = Taxonomy.from_test_json(paths.test_json)
    sources = [paths.test_json] if shot == 0 else sorted(paths.few_shot_dir.glob(f"camo5_*_{shot}shot_split1.json"))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    report = audit_test(paths, taxonomy) if shot == 0 else audit_shot(shot, paths, taxonomy)
    assert not report.errors, report.errors
    assert report.annotations, "Geometry preflight must actually convert annotations"
    annotations, images = deepcopy(report.annotations), deepcopy(report.images)
    for annotation in report.annotations:
        line, _ = annotation_to_yolo(annotation, report.images[annotation["image_id"]], taxonomy.category_to_index)
        assert all(0 <= float(value) <= 1 for value in line.split()[1:])
    assert report.annotations == annotations and report.images == images
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources} == before
    assert not paths.work_root.exists()
