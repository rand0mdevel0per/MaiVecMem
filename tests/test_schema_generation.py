import os
import ujson
from plugins.MaiVecMem import cli_tool

# Simple test: create a fake model_info.json with dimension 512 and ensure schema.generated.sql is created

def test_schema_generation(tmp_path, monkeypatch):
    os.path.dirname(os.path.abspath(__file__))

    # create fake model_info.json in plugin dir
    mi = {
        "model": "fake-model",
        "dimension": 512,
        "plugin_config_snapshot": {"openai_embedding": {"model": "fake-model", "base_url": "http://fake"}},
        "global_config_excerpt": {},
    }
    mi_path = os.path.join(os.path.dirname(__file__), "..", "model_info.json")
    with open(mi_path, "w", encoding="utf-8") as f:
        ujson.dump(mi, f)

    # call initialize_database but we need a fake asyncpg connection that has transaction method
    class FakeConn:
        def transaction(self):
            class Dummy:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

            return Dummy()

        async def execute(self, script):
            # write to a file to simulate execution
            open(os.path.join(os.path.dirname(__file__), "..", "executed_schema.sql"), "w", encoding="utf-8").write(script)

    fake_conn = FakeConn()

    import asyncio

    asyncio.run(cli_tool.initialize_database(fake_conn, cfg=None))

    gen_path = os.path.join(os.path.dirname(__file__), "..", "schema.generated.sql")
    assert os.path.exists(gen_path)

