"""HTML page: V1 vs V2 (language policy + LLM glossary + polish preview), one card per unit.
usage: python build_v2_compare.py <v2_preview.json> <aldi_dialectness.json | -> <out.html>"""
import html
import json
import sys
from pathlib import Path

PREVIEW, ALDI, OUT = sys.argv[1:4]
d = json.loads(Path(PREVIEW).read_text(encoding='utf-8'))
aldi = json.loads(Path(ALDI).read_text(encoding='utf-8')) if ALDI != '-' else None
terms = {t['id']: t for t in d['glossary']['terms']}
CALQUE_AR = {'yaqum_bi': '«يقوم بـ + مصدر»', 'yatadih_annahu': '«يتضح أنه»', 'ma_sanafaluhu': '«ما سنفعله هو»',
             'ibara_an': '«عبارة عن»', 'maa_badiha': '«مع بعضها»', 'yatahassal': '«يتحصّل»'}


def esc(x):
    return html.escape(x or '')


def badges(f):
    out = []
    for t in f.get('terms', []):
        ar = terms.get(t, {}).get('ar', t)
        out.append(f"<span class='b term'>مصطلح: {esc(ar)}</span>")
    for c in f.get('calques', []):
        out.append(f"<span class='b calq'>صياغة حرفية: {esc(CALQUE_AR.get(c, c))}</span>")
    if f.get('dialect'):
        out.append(f"<span class='b dia'>عامية: {esc('، '.join(f['dialect']))}</span>")
    return ''.join(out) or "<span class='b ok'>سليم</span>"


S = d['summary']
rows = [('عدد الجمل', S['before_v1']['units'], S['after_v2_select_plus_polish_preview']['units']),
        ('جمل فيها مصطلح مخالف للقاموس', S['before_v1']['term_missing'], S['after_v2_select_plus_polish_preview']['term_missing']),
        ('جمل فيها صياغة حرفية', S['before_v1']['stiff'], S['after_v2_select_plus_polish_preview']['stiff']),
        ('جمل فيها كلمات عامية', S['before_v1']['dialect'], S['after_v2_select_plus_polish_preview']['dialect'])]
if aldi:
    rows.append(('درجة اللهجة ALDi (0 فصحى، 1 عامية) — المتوسط', aldi['lecture_v1']['mean'], aldi['lecture_v2_preview']['mean']))
summary = ''.join(f"<tr><td>{a}</td><td class='num'>{b}</td><td class='num'>{c}</td></tr>" for a, b, c in rows)
gl = ''.join(f"<tr><td dir='ltr'>{esc(', '.join(e for e in t['en'] if not e.startswith('re:')) or t['id'])}</td>"
             f"<td>{esc(t['ar'])}</td><td>{esc('، '.join(t['avoid']))}</td></tr>" for t in d['glossary']['terms'])

cards = []
for u in d['units']:
    v1, sel, pol = u['v1'], u['v2_select'] or {}, u['v2_polish_preview']
    v2_text = (pol or sel or v1)['text']
    v2_f = (pol or {}).get('findings') or sel.get('findings') or v1['findings']
    changed = v2_text != v1['text']
    how = 'polish' if pol else ('select' if changed else 'same')
    how_ar = {'polish': 'تحرير لغوي (معاينة)', 'select': 'ترجمة أخرى من المرشحين', 'same': 'بلا تغيير'}[how]
    a = aldi['per_unit'].get(u['unit_id']) if aldi else None
    cards.append(
        f"<section class='unit {'changed' if changed else ''}' id='{u['unit_id']}'>"
        f"<h2>{u['unit_id']} <span class='meta'>{how_ar}</span></h2>"
        f"<p class='en' dir='ltr'>{esc(u['source'])}</p>"
        f"<div class='pair'><div class='col'><h3>قبل (V1)</h3><p class='ar'>{esc(v1['text'])}</p>{badges(v1['findings'])}"
        + (f"<p class='mut'>ALDi {a[0]}</p>" if a else '') +
        f"</div><div class='col v2'><h3>بعد (V2)</h3><p class='ar'>{esc(v2_text)}</p>{badges(v2_f)}"
        + (f"<p class='mut'>ALDi {a[1]}</p>" if a else '') + "</div></div></section>")

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6b66;--line:#e3e0d8;--card:#fff;--v2:#f1f8f1;--term:#fde7e7;--calq:#fff1d6;--ok:#e6f4ea;--dia:#ead9ff}
@media (prefers-color-scheme:dark){:root{--bg:#151514;--fg:#eceae4;--mut:#a3a19a;--line:#33322f;--card:#1d1d1b;--v2:#18261b;--term:#3a1f1f;--calq:#3a3020;--ok:#1d3324;--dia:#2c2140}}
body{background:var(--bg);color:var(--fg);font:15px/1.75 system-ui,"Segoe UI",Tahoma,sans-serif;margin:0 auto;padding:16px;max-width:1200px}
h1{font-size:22px;margin:8px 0} h2{font-size:15px;margin:0 0 6px} h3{font-size:13px;margin:0 0 4px;color:var(--mut)}
.meta,.mut{color:var(--mut);font-weight:400;font-size:13px}
.unit,.box{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:14px 0}
.unit.changed{border-color:#7fb08a}
.en{background:var(--bg);border-radius:6px;padding:6px 10px;margin:4px 0 8px;text-align:left}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px} .col{padding:8px;border-radius:8px} .col.v2{background:var(--v2)}
.ar{margin:0 0 6px} .b{display:inline-block;font-size:12px;border-radius:10px;padding:1px 8px;margin:2px 0 0 4px}
.b.term{background:var(--term)} .b.calq{background:var(--calq)} .b.ok{background:var(--ok)} .b.dia{background:var(--dia)}
table{border-collapse:collapse;width:100%;font-size:14px} th,td{border-top:1px solid var(--line);padding:6px;text-align:right;vertical-align:top}
th{color:var(--mut);font-size:12px} td.num{text-align:center;direction:ltr} .wrap{overflow-x:auto}
@media (max-width:700px){.pair{grid-template-columns:1fr}}
"""
page = ("<!doctype html><html lang='ar' dir='rtl'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'><title>قبل وبعد V2</title>"
        f"<style>{CSS}</style></head><body>"
        "<h1>ترجمة المحاضرة: قبل (V1) وبعد (V2)</h1>"
        f"<p class='mut'>التشغيل {esc(d.get('run_id'))} · القاموس بناه الموديل من نص الفيديو كاملًا "
        f"({len(d['glossary']['terms'])} مصطلحًا) · سياسة اللغة {esc(d['policy'])}</p>"
        "<div class='box'><b>الملخص</b><div class='wrap'><table><tr><th>المقياس</th><th>قبل</th><th>بعد</th></tr>"
        f"{summary}</table></div>"
        "<p class='mut'>تنبيه: «بعد» هنا معاينة. التحرير اللغوي لم يُفحص معناه بـ MetricX بعد (يحتاج GPU)، "
        "والتشغيل الكامل لـ V2 بالموديلات مع القاموس يكون على Colab. بعض «المصطلحات المخالفة» قبلُ سببها أن "
        "القاموس الجديد اختار كلمة مختلفة (مثل «كثافة» بدل «شدة»)، لذلك يحتاج القاموس مراجعة بشرية مرة واحدة.</p></div>"
        "<details class='box'><summary><b>القاموس الذي بناه الموديل</b></summary><div class='wrap'><table>"
        f"<tr><th>English</th><th>المعتمد</th><th>نتجنّب</th></tr>{gl}</table></div></details>"
        + ''.join(cards) + "</body></html>")
Path(OUT).write_text(page, encoding='utf-8')
print('written', OUT)
