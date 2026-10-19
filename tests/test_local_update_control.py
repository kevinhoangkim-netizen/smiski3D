from collections import OrderedDict
import json
import socket
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.app import main
from scripts import launch_local


def test_update_stop_is_local_authenticated_and_does_not_expose_token(monkeypatch):
    token='private-test-token-123456789012345'
    monkeypatch.setenv('LITTLE_FIGURES_UPDATE_TOKEN',token)
    monkeypatch.setattr(main,'jobs',OrderedDict())
    scheduled=[]
    monkeypatch.setattr(main,'schedule_local_shutdown',lambda:scheduled.append(True))
    local=TestClient(main.app,client=('127.0.0.1',12345))
    remote=TestClient(main.app,client=('192.0.2.1',12345))
    headers={'X-Local-Update-Token':token}
    try:
        assert local.post('/api/local-update/stop').status_code==403
        assert remote.post('/api/local-update/stop',headers=headers).status_code==403
        health=local.get('/api/health')
        assert token not in health.text
        response=local.post('/api/local-update/stop',headers=headers)
        assert response.status_code==200 and response.json()['status']=='stopping'
        assert scheduled==[True] and token not in response.text
        assert local.post('/api/project/check',json={'figures':[]}).status_code==503
    finally:
        main.local_update_pending.clear()


def test_update_stop_waits_for_generation_and_analysis(monkeypatch):
    monkeypatch.setenv('LITTLE_FIGURES_UPDATE_TOKEN','local-test-token')
    monkeypatch.setattr(main,'jobs',OrderedDict(one={'state':'generating'}))
    client=TestClient(main.app,client=('127.0.0.1',12345))
    headers={'X-Local-Update-Token':'local-test-token'}
    assert client.post('/api/local-update/stop',headers=headers).status_code==409
    main.jobs.clear()
    with main.analysis_slot:
        assert client.post('/api/local-update/stop',headers=headers).status_code==409
    assert not main.local_update_pending.is_set()


def test_launcher_only_cleans_its_own_runtime_record(tmp_path,monkeypatch):
    monkeypatch.setattr(launch_local,'LOCAL',tmp_path)
    monkeypatch.setattr(launch_local,'ROOT',tmp_path)
    launch_local.write_runtime(SimpleNamespace(pid=1234),8765,'first-secret')
    record=json.loads((tmp_path/'runtime.json').read_text())
    assert record['pid']==1234 and record['port']==8765
    assert record['root']==str(tmp_path.resolve())
    assert record['version']==main.APP_VERSION
    launch_local.clear_runtime(1234,'other-secret')
    assert (tmp_path/'runtime.json').exists()
    launch_local.clear_runtime(1234,'first-secret')
    assert not (tmp_path/'runtime.json').exists()


def test_launcher_skips_active_listener_but_reuses_a_recently_closed_port():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind(('127.0.0.1',0));server.listen()
        port=server.getsockname()[1]
        assert launch_local.choose_port(port)!=port
        connection=socket.create_connection(('127.0.0.1',port))
        accepted,_=server.accept()
        accepted.close();connection.close()
    assert launch_local.choose_port(port)==port
