"""CLI for preparing work from existing artifacts, with safe defaults.

python -m arabic_dubbing_v2.cli --help
"""
import argparse
import json
from pathlib import Path
from .common import read_json, save_new_json
from .boundaries import build_units_from_whisperx
from .workflow import prepare_project
from .translation import build_all_pass_plans
from .speech import run_edge_previews
from .terminology import validation_plans


def main(argv=None):
    p=argparse.ArgumentParser(description='Arabic Dubbing V2 (offline by default)')
    sub=p.add_subparsers(dest='cmd',required=True)
    b=sub.add_parser('build',help='Build conservative TUs from saved WhisperX aligned JSON (experimental)')
    b.add_argument('--aligned',required=True)
    b.add_argument('--out',required=True)
    b.add_argument('--course-title',default=None)
    b.add_argument('--course-id',default=None)
    prep=sub.add_parser('prepare',help='Enrich existing master with dynamic glossary, context and QA')
    prep.add_argument('--master',required=True)
    prep.add_argument('--out',required=True)
    for k in ('course-id','course-title','chapter','lesson','topic','description'):
        prep.add_argument('--'+k,default=None)
    plans=sub.add_parser('plan-translation',help='Export all four prompt-stage plans. NO calls.')
    plans.add_argument('--master',required=True)
    plans.add_argument('--out',required=True)
    term=sub.add_parser('plan-terminology',help='Export contextual terminology-validation plans. NO calls.')
    term.add_argument('--master',required=True)
    term.add_argument('--out',required=True)
    v=sub.add_parser('preview',help='Edge TTS. Plan only unless --execute --allow-online-tts both set.')
    v.add_argument('--master',required=True)
    v.add_argument('--out-dir',required=True)
    v.add_argument('--units',nargs='*',default=None)
    v.add_argument('--voice',default='ar-SA-HamedNeural')
    v.add_argument('--rate',default='+0%')
    v.add_argument('--execute',action='store_true')
    v.add_argument('--allow-online-tts',action='store_true')
    a=p.parse_args(argv)
    if a.cmd=='build':
        raw=read_json(a.aligned)
        source=raw.get('source_segments',raw.get('segments'))
        result=build_units_from_whisperx(source,course={'id':a.course_id,'title':a.course_title})
        save_new_json(a.out,result)
        message={'built_units':len(result['translation_units']), 'word_coverage':result['builder_report']['source_word_coverage_verified'], 'source_unchanged':True}
    elif a.cmd=='prepare':
        overrides={k.replace('-','_'):getattr(a,k.replace('-','_')) for k in ('course-id','course-title','chapter','lesson','topic','description')}
        result=prepare_project(a.master,a.out,course_overrides={'id':overrides.pop('course_id'),'title':overrides.pop('course_title'),**overrides})
        message=result['translation_v2_report']
    elif a.cmd=='plan-translation':
        message=build_all_pass_plans(read_json(a.master))
        save_new_json(a.out,message)
        message={'status':message['status'],'planned_units':len(message['plans']),'out':a.out}
    elif a.cmd=='plan-terminology':
        prepared=read_json(a.master)
        message=validation_plans(prepared['terminology_v2'],prepared['translation_v2_metadata']['course'])
        save_new_json(a.out,message)
        message={'status':message['status'],'planned_terms':len(message['plans']),'out':a.out}
    elif a.cmd=='preview':
        message=run_edge_previews(read_json(a.master),a.out_dir,unit_ids=a.units,
                                  voice=a.voice,rate=a.rate,execute=a.execute,
                                  allow_online_tts=a.allow_online_tts)
    print(json.dumps(message,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
