from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent
m=json.loads((R/'EXPORT_MANIFEST.json').read_text())
for entry in m['files']:
 p=R/entry['path'];assert p.is_file(),p
 assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],p
s=json.loads((R/'results_summary.json').read_text())
assert s['S13']['assisted_task_complete'] and not s['S13']['autonomous_end_to_end']
assert s['S16']['assisted_task_complete'] and not s['S16']['strict_quality_complete']
assert not any(s['S22']['autonomous_end_to_end'])
for tag,expected in [('S13',True),('S16',True),('S0_n10',True),('S22_0500',False),('S22_1000',False),('S22_2000',False)]:
 e=json.loads((R/'evidence'/f'{tag}_RESULT.json').read_text());assert e['full_task_success']==expected,tag
print('SHA and recorded outcome check PASS:',len(m['files']),'files')
print(json.dumps(s,ensure_ascii=False,indent=2))
print('This verifies exported records, not a new robot rollout.')
