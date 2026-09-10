from __future__ import annotations

from types import SimpleNamespace

from backend.app.providers.baostock_provider import BaoStockProvider


class FakeBaoStock:
    def __init__(self):
        self.login_calls = 0
        self.query_calls = 0

    def login(self):
        self.login_calls += 1
        return SimpleNamespace(error_code="0", error_msg="success")

    def logout(self):
        return SimpleNamespace(error_code="0", error_msg="success")

    def query_history_k_data_plus(self, *_args, **_kwargs):
        self.query_calls += 1
        if self.query_calls == 1:
            return SimpleNamespace(error_code="10001001", error_msg="用户未登录")
        return SimpleNamespace(error_code="0", error_msg="success", fields=[], next=lambda: False)


def test_long_batch_relogs_and_retries_expired_session():
    fake = FakeBaoStock()
    provider = BaoStockProvider.__new__(BaoStockProvider)
    provider.bs = fake
    provider._entered = True
    response = provider._retry_after_relogin(lambda: fake.query_history_k_data_plus("code", "fields"))
    assert response.error_code == "0"
    assert fake.login_calls == 1
    assert fake.query_calls == 2
