import json
import os
import yaml

name = {{ jinx_name | tojson }}
desc = {{ description | tojson }}
inputs_raw = {{ inputs_spec | tojson }}
body = {{ python_code | tojson }}
subdir = {{ target_subdir | tojson }}

try:
    inputs_list = json.loads(inputs_raw) if isinstance(inputs_raw, str) else inputs_raw
except (json.JSONDecodeError, TypeError):
    inputs_list = []
if not isinstance(inputs_list, list):
    inputs_list = []

team = getattr(npc, 'team', None) if npc else None
if team is None and state is not None:
    team = getattr(state, 'team', None)
if team is None or not getattr(team, 'team_path', None):
    context['output'] = json.dumps({"error": "No team available to host the new jinx."})
else:
    jinx_dir = os.path.join(team.team_path, 'jinxes', subdir)
    os.makedirs(jinx_dir, exist_ok=True)
    out_path = os.path.join(jinx_dir, name + '.jinx')

    jinx_yaml = {
        'jinx_name': name,
        'description': desc,
        'inputs': [{i: ''} if isinstance(i, str) else i for i in inputs_list],
        'steps': [
            {'name': 'run', 'engine': 'python', 'code': body.rstrip() + '\n'}
        ],
    }

    with open(out_path, 'w') as f:
        yaml.safe_dump(jinx_yaml, f, sort_keys=False, default_flow_style=False)

    registered = False
    try:
        from npcpy.npc_compiler import Jinx
        new_jinx = Jinx(jinx_path=out_path)
        team.jinxes_dict[name] = new_jinx
        if hasattr(team, '_raw_jinxes_list'):
            team._raw_jinxes_list.append(new_jinx)
        registered = True
    except Exception as e:
        context['_register_error'] = str(e)

    context['output'] = json.dumps({
        "path": out_path,
        "name": name,
        "registered": registered,
    })
