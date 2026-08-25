import numpy as np
from engine import simulate
CORR=[[1,.68,.30,.10,0],[.68,1,.80,.30,0],[.30,.80,1,.25,0],[.10,.30,.25,1,0],[0,0,0,0,1]]
W=[45,20,10,8,17]; V=[7.02,14.47,19.80,18.30,0.67]
EXCEL={ # workbook values, Base
 3:dict(p50=7.589,pl=4.79,wm=1.2454),5:dict(p50=7.470,pl=1.32,wm=1.4336),10:dict(p50=7.517,pl=0.09,wm=2.0643)}
for label,eqr in (("BASE",12.88),("CONS",7.00)):
    R=[5.00,eqr,11.90,6.10,3.80]
    res,rep=simulate(W,R,V,CORR,10000,10,20260722)
    print(f"\n== {label}  {res.elapsed_ms:.0f} ms  repaired={rep}")
    print(f"   analytic arith {res.exp_arith*100:.2f}  analytic vol {res.analytic_vol*100:.2f}  sim vol {res.sim_vol*100:.2f}")
    for r in res.rows:
        print(f"   {r['_years']:>3}y  p5 {r['5th']*100:7.2f}  p50 {r['Median']*100:7.2f}  p95 {r['95th']*100:7.2f}"
              f"  mean {r['Mean']*100:6.2f}  P(loss) {r['P(loss)']*100:6.2f}  wMed {r['Median x']:.4f}  wP5 {r['5th x']:.4f}  wP95 {r['95th x']:.4f}")
        if label=="BASE":
            e=EXCEL[r['_years']]
            print(f"        excel: p50 {e['p50']:.3f}  P(loss) {e['pl']:.2f}  wMed {e['wm']:.4f}"
                  f"   -> diff p50 {abs(r['Median']*100-e['p50']):.3f}pp, wMed {abs(r['Median x']-e['wm']):.4f}")
# seed stability + non-PSD repair
a,_=simulate(W,[5,12.88,11.9,6.1,3.8],V,CORR,5000,10,7); b,_=simulate(W,[5,12.88,11.9,6.1,3.8],V,CORR,5000,10,7)
print("\nseed reproducible:", a.rows[-1]['Median x']==b.rows[-1]['Median x'])
bad=[[1,.9,-.9,0,0],[.9,1,.9,0,0],[-.9,.9,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
r,rep=simulate(W,[5,12.88,11.9,6.1,3.8],V,bad,2000,5,1)
print("non-PSD matrix handled:", rep, "| finite:", np.isfinite(r.rows[-1]['Median x']))
