import json


async def test_health(jp_fetch):
    # When
    response = await jp_fetch("jupyter-ai-persona-manager", "health")

    # Then
    assert response.code == 200
    payload = json.loads(response.body)
    assert payload == {
        "data": "This is /jupyter-ai-persona-manager/get-example endpoint!"
    }