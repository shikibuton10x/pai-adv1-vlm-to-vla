"""S15 local file conversion only; no physics/model/network."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.accepted_data_s15 import export
if __name__=='__main__':
    m=export(ROOT/'data/s15_single_accepted_v1');print(json.dumps(dict(total=m['total'],episodes=len(m['episodes']))))
