import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_s14_center as core
from run_s14_clarify import config,add_instruction,installed_clarification,validate_start,check_start
from course_award.center_restart_s14r import RestartLedger
from course_award.budget import BudgetStop
class Clarification(unittest.TestCase):
    def test_exact_contract_text(self):
        expected=next(x[2:] for x in (ROOT/'docs/S14C_VISUAL_CENTER_FOLLOWUP.md').read_text().splitlines() if x.startswith('> Additional human instruction:'))
        self.assertEqual(config()['human_instruction'],expected)
    def test_complete_initial_wire_only_SYSTEM_suffix_and_payload_unchanged(self):
        old=json.loads((ROOT/'results/s14r_ready_v1/UNSENT_REQUEST.json').read_text());payload=json.loads((ROOT/'results/s14r_ready_v1/PAYLOAD_000.json').read_text());before=copy.deepcopy(old)
        p,new=add_instruction(payload,old);self.assertIs(p,payload);expected=copy.deepcopy(old);expected['systemInstruction']['parts'][0]['text']+='\n\n'+config()['human_instruction'];self.assertEqual(new,expected);self.assertEqual(old,before)
    def test_noninitial_builder_adds_once_and_restores_physics_and_globals(self):
        request=dict(systemInstruction=dict(parts=[dict(text='Existing SYSTEM')]),contents=[dict(parts=[dict(text='Robot-only recent history')])],generationConfig=dict(maxOutputTokens=32768,thinkingConfig=dict(thinkingLevel='MEDIUM')))
        objects={k:getattr(core,k) for k in ['Executor','Recorder','S14Stability','CenterGeometry','make_request','snapshot','write','config','VERSION','PilotLedger','Ledger']}
        with patch.object(core,'make_request',return_value=({'history':['robot/action only']},request)):
            with installed_clarification():
                for k in ['Executor','Recorder','S14Stability','CenterGeometry']:self.assertIs(getattr(core,k),objects[k])
                for n in range(2):
                    p,r=core.make_request('noninitial');self.assertEqual(r['systemInstruction']['parts'][0]['text'],'Existing SYSTEM\n\n'+config()['human_instruction']);self.assertEqual(p,{'history':['robot/action only']})
                self.assertEqual(request['systemInstruction']['parts'][0]['text'],'Existing SYSTEM')
        for k,v in objects.items():self.assertIs(getattr(core,k),v)
    def test_shared_pool_used8_remaining22_and_old402_retained(self):
        v=validate_start();self.assertEqual(v['remaining_pool_generations'],22);self.assertEqual(v['remaining_pool_micro'],862210);self.assertEqual(v['terminal_unknown_charge_max_micro'],345680)
        with installed_clarification():
            self.assertIs(core.PilotLedger,RestartLedger);self.assertEqual(core.config()['generation_cap'],30);self.assertEqual(core.config()['phase_money_cap_micro'],1000000)
    def test_new_pending_or_baseline_drift_stops_without_writing(self):
        l=RestartLedger(ROOT/'results/S1_API_LEDGER.jsonl');raw=l.summary();scope=l.scoped(raw);changed=copy.deepcopy(raw);changed['pending_ids'].append('newunknown');changed['entries']['newunknown']=dict(pending=True,reserve_micro=345680);changed['committed_micro']+=345680
        with self.assertRaises(BudgetStop):check_start(changed,scope,config())
        changed=copy.deepcopy(raw);changed['generations']+=1
        with self.assertRaises(BudgetStop):check_start(changed,scope,config())
if __name__=='__main__':unittest.main()
