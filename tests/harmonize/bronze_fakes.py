"""CSV builders shared by the bronze tests. A module of its own (not conftest.py), because the repo has
several conftest.py files and `from conftest import ...` resolves to whichever was loaded first."""

ROWS = [("015009", "Oak Manor"), ("15E123", "Pine Court")]
HEADER = '"CMS Certification Number (CCN)","Provider Name","Processing Date"\r\n'


def csv_text(rows, *, blank_tail=False, processing="2025-06-01"):
    body = "".join(f'"{ccn}","{name}","{processing}"\r\n' for ccn, name in rows)
    return HEADER + body + ("\r\n" if blank_tail else "")
