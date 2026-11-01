"""Convert already saved S8 RAW video locally. No sim/model/API calls."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.cabinet_recording_s8 import finish_video
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Existing dedicated S8 result only')
    expected=sum(1 for _ in (out/'BOUNDARY_INDEX.jsonl').open())
    print(json.dumps(finish_video(out,expected)))
if __name__=='__main__':main()
