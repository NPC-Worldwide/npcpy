import json
import os

ref = {{ jinx_ref | tojson }}
values_raw = {{ input_values | tojson }}

try:
    values = json.loads(values_raw) if isinstance(values_raw, str) else values_raw
except (json.JSONDecodeError, TypeError):
    values = {}
if not isinstance(values, dict):
    values = {}

team = getattr(npc, 'team', None) if npc else None
if team is None and state is not None:
    team = getattr(state, 'team', None)

target = None
if team is not None and hasattr(team, 'jinxes_dict') and ref in team.jinxes_dict:
    target = team.jinxes_dict[ref]
elif os.path.exists(ref):
    try:
        from npcpy.npc_compiler import Jinx
        target = Jinx(jinx_path=ref)
    except Exception as e:
        context['output'] = json.dumps({"error": "Failed to load jinx from path: " + str(e)})
        target = None

if target is None:
    if 'output' not in context:
        context['output'] = json.dumps({
            "error": "Jinx '" + ref + "' not found on team and no file at that path.",
        })
else:
    try:
        result = target.execute(
            input_values=values,
            npc=npc,
            messages=list(messages) if messages else [],
        )
        out = result.get('output') if isinstance(result, dict) else result
        context['output'] = out if isinstance(out, str) else json.dumps(out, default=str)
    except Exception as e:
        context['output'] = json.dumps({"error": "Execution failed: " + str(e)})
