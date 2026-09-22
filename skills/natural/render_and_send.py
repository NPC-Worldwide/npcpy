from jinja2 import Template
tpl = Template({{ prompt | tojson }})
rendered = tpl.render(context)
system_msg = {{ system | tojson }}
messages = []
if system_msg:
    messages.append({"role": "system", "content": system_msg})
response = npc.get_llm_response(rendered, messages=messages)
if isinstance(response, dict):
    output = response.get("response", response)
else:
    output = response
