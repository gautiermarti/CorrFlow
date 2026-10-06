import sys

import pytest

from corrflow.cli.evaluate_flow import main


@pytest.mark.parametrize("existing", ["evaluation.json", "evaluation.npz"])
def test_evaluation_does_not_replace_prior_results(tmp_path, monkeypatch, existing):
    saved = tmp_path / existing
    saved.write_bytes(b"prior experiment")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--data",
            "missing",
            "--checkpoint",
            "missing.pt",
            "--output",
            str(tmp_path / "evaluation.json"),
        ],
    )
    with pytest.raises(FileExistsError):
        main()
    assert saved.read_bytes() == b"prior experiment"
    assert len(list(tmp_path.iterdir())) == 1
