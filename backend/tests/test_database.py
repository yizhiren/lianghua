from __future__ import annotations

import sqlite3

import pytest

from backend.app.database import Database


def test_failed_write_rolls_back_instead_of_leaving_database_locked(tmp_path):
    database = Database(tmp_path / "rollback.db")
    database.initialize()
    values = ("SH.600000", "SH", "600000", "浦发银行", "CNY")
    database.execute(
        "INSERT INTO securities(id,market,code,name,currency) VALUES(?,?,?,?,?)",
        values,
    )

    with pytest.raises(sqlite3.IntegrityError):
        database.execute(
            "INSERT INTO securities(id,market,code,name,currency) VALUES(?,?,?,?,?)",
            values,
        )

    assert database.connection().in_transaction is False
    database.execute(
        "INSERT INTO securities(id,market,code,name,currency) VALUES(?,?,?,?,?)",
        ("SZ.000001", "SZ", "000001", "平安银行", "CNY"),
    )
    assert database.one("SELECT 1 FROM securities WHERE id='SZ.000001'") is not None
