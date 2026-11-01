import ast,json,unittest
from pathlib import Path
import numpy as np
from course_award.sft_data_s11 import TeacherData,chunk_indices,FIELDS,sha
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/s15_single_accepted_v1'
class SingleDemo(unittest.TestCase):
 def test_one1028_source_actors_and_observer_exclusion(self):
  from course_award.accepted_data_s15 import source
  p,idx,z=source();d=TeacherData(DATA);self.assertEqual(d.total,1028);self.assertEqual(d.lengths,[1028]);self.assertEqual(d.episodes[0]['actions'].tobytes(),z['actions'][:1028].tobytes());self.assertTrue(idx[1028]['incoming_action_teacher_label']);self.assertFalse(idx[1029]['incoming_action_teacher_label'])
 def test_last_chunk_clamps_and_masks_before_observer(self):
  ids,pad=chunk_indices(1027,1028);self.assertTrue(np.all(ids==1027));self.assertEqual(np.count_nonzero(pad),49);self.assertFalse(pad[0]);ids,pad=chunk_indices(1007,1028);self.assertLess(ids.max(),1028);self.assertEqual(np.count_nonzero(pad),29)
 def test_eight_independent_original_observation_pairs(self):
  d=TeacherData(DATA);probes=json.loads((ROOT/'fixtures/s15_processor_inputs/PROVENANCE.json').read_text());self.assertEqual(list(map(int,probes)),[0,128,350,799,906,943,963,1027])
  for key,v in probes.items():
   f=ROOT/v['fixture'];self.assertEqual(sha(f),v['original_sha256']);self.assertEqual(sha(ROOT/v['source_boundary']),v['source_boundary_sha256'])
   with np.load(f) as original:
    for k,raw in FIELDS.items():self.assertEqual(d.episodes[0][k][int(key)].tobytes(),original[raw].tobytes())
 def test_original_math_and_processor_helpers_unchanged_and_adoption_separate(self):
  a=ast.parse((ROOT/'scripts/train_s11_small.py').read_text());b=ast.parse((ROOT/'scripts/train_s15_single.py').read_text())
  for name in ['learning_rate','parity_fixture_batch','update','tensor_hashes']:
   get=lambda x:next(n for n in ast.walk(x) if isinstance(n,ast.FunctionDef) and n.name==name)
   self.assertEqual(ast.dump(get(a)),ast.dump(get(b)))
  adopted=json.loads((DATA/'HUMAN_ADOPTION.json').read_text());self.assertTrue(adopted['accepted']);self.assertFalse(adopted['original_center20mm_success']);raw=json.loads((ROOT/'results/s14d_approach_center_live_v1/RESULT.json').read_text());self.assertFalse(raw['full_center_demonstration_success']);self.assertFalse(raw['human_quality_accepted'])
if __name__=='__main__':unittest.main()
