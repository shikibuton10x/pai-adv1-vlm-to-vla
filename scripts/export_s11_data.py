"""Local file conversion only. No physics, inference, API, package installation."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.sft_data_s11 import export

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'data/s11_teacher_pairs_v1');a=p.parse_args()
    out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated project output only')
    m=export(ROOT,out);print(json.dumps({'episodes':[(e['id'],e['rows']) for e in m['episodes']],'total':m['total'],'join872':m['join872']}))
