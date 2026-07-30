"""Figure 1, conceptual overview for the UKB wrist-actigraphy disease-prediction paper.

Phenome breadth plus orthogonal genetics on a single left-to-right spine, drawn with
Tabler line-icons (MIT) as vector paths. Iconographic; the only real data is the
Parkinson's lead-time inset. Draft written to fig1_assets/_fig1_draft.{png,pdf}.

Expects the figure assets (SVG icons, the vendored svgpath2mpl parser) under a
fig1_assets/ directory beside the paper root; set UKB_ROOT for the output location.
"""
import sys, re, math, os
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent.parent / "fig1_assets"
sys.path.insert(0, str(_ASSETS / "pylibs"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Wedge, FancyBboxPatch, Circle
from matplotlib.transforms import Affine2D
from svgpath2mpl import parse_path

# palette
C_WRIST="#0072B2"; C_GEN="#CC79A7"; C_NEURO="#D55E00"; C_MORT="#000000"
C_SPINE="#4D4D4D"; C_SUB="#7F7F7F"; INK="#1a1a1a"
CAT_COLORS={"Cardiovascular":"#332288","Neurological":C_NEURO,"Genitourinary":"#6699CC",
 "Endocrine/Metab":"#DDCC77","Mental":"#44AA99","Musculoskeletal":"#999933",
 "Respiratory":"#88CCEE","Sense organs":"#CC6677","Dermatological":"#EE8866",
 "Neoplasms":"#882255","Blood/Immune":"#661100","Gastrointestinal":"#117733"}
ICON=str(_ASSETS / "icons")

def _ipaths(name):
    s=open(f"{ICON}/{name}.svg").read()
    return [parse_path(d) for d in re.findall(r'\sd="([^"]+)"', s) if d.strip()!="M0 0h24v24H0z"]
def icon(ax,name,x,y,size,color=INK,lw=1.5,z=6):
    sc=size/24.0; t=Affine2D().scale(sc,-sc).translate(x-size/2,y+size/2)
    for p in _ipaths(name):
        ax.add_patch(PathPatch(p.transformed(t),fill=False,edgecolor=color,lw=lw,
                               joinstyle="round",capstyle="round",zorder=z))

plt.rcParams.update({"font.family":"sans-serif",
    "font.sans-serif":["Arimo","Arial","Helvetica","DejaVu Sans"],"pdf.fonttype":42})
fig=plt.figure(figsize=(180/25.4,75/25.4))
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,180); ax.set_ylim(22,97)
ax.set_aspect("equal"); ax.axis("off")
def T(x,y,s,size=7,color=INK,ha="center",va="center",weight="normal",style="normal",rot=0,z=8):
    ax.text(x,y,s,fontsize=size,color=color,ha=ha,va=va,fontweight=weight,fontstyle=style,
            rotation=rot,zorder=z,linespacing=1.08)
def arrow(p0,p1,color,lw,z=5,ls="-",head=True):
    ax.annotate("",xy=p1,xytext=p0,zorder=z,arrowprops=dict(
        arrowstyle="-|>" if head else "-",color=color,lw=lw,ls=ls,shrinkA=0,shrinkB=0,mutation_scale=9))

SP=45  # spine height
# ===================== SPINE (two segments, around the arc) =====================
arrow((28,SP),(57,SP),C_SPINE,1.25,z=2)          # zone2 -> arc
arrow((103,SP),(151,SP),C_SPINE,1.25,z=2)        # arc -> zone4

# ===================== ZONE 1: INPUT =====================
T(16,73,"One week of\nwrist movement",size=6.9,weight="bold")
icon(ax,"device-watch",16,55,14,color=C_WRIST,lw=1.9)
xx=np.linspace(9,23,80)
for k,off in enumerate([0,-1.6,-3.2]):
    ax.plot(xx,44+off+0.9*np.sin((xx-9)*2.1+k*1.9)*np.exp(-((xx-16)**2)/95),
            color=C_WRIST,lw=0.7,alpha=0.85,zorder=5)
T(16,31,"Axivity AX3",size=5.7,color=C_SUB)
T(16,27.8,"~7 days",size=5.7,color=C_SUB)

# ===================== ZONE 2: MODEL =====================
T(40,73,"Frozen\nfoundation\nmodel",size=6.9,weight="bold")
icon(ax,"filter",39.5,54,13,color=INK,lw=1.8)
icon(ax,"lock",46,59,5.6,color=C_SPINE,lw=1.35)
T(40,31,"weights frozen",size=5.7,color=C_SUB)
T(40,27.8,"108,904 recordings",size=5.7,color=C_SUB)
tx=53
ax.add_patch(FancyBboxPatch((tx-3.6,SP-2.4),7.2,4.8,boxstyle="round,pad=0.2,rounding_size=1.2",
    fc="white",ec=C_SPINE,lw=0.9,zorder=7))
for i,cc in enumerate([C_WRIST,C_NEURO,C_GEN,"#117733"]):
    ax.add_patch(Circle((tx-2.0+i*1.35,SP),0.5,color=cc,zorder=8))

# ===================== ZONE 3: PHENOME ARC (primary hero) =====================
CATS=[("Cardiovascular",0.733),("Neurological",0.729),("Genitourinary",0.728),
 ("Endocrine/Metab",0.724),("Mental",0.720),("Musculoskeletal",0.670),("Respiratory",0.669),
 ("Sense organs",0.665),("Dermatological",0.657),("Neoplasms",0.649),("Blood/Immune",0.647),
 ("Gastrointestinal",0.643)]
order=sorted(CATS,key=lambda kv:-kv[1]); n=len(order); center=(n-1)/2
arr=[None]*n
for rank,p in enumerate(sorted(range(n),key=lambda p:abs(p-center))): arr[p]=order[rank]
FX,FY=80,SP; R0=6.5; K=64.0; SPAN=172; A0=(180-SPAN)/2
def rr(c): return R0+(c-0.5)*K
for i,(cat,c) in enumerate(arr):
    t1=A0+i*SPAN/n; t2=A0+(i+1)*SPAN/n; ro=rr(c)
    if cat=="Neurological":
        ax.add_patch(Wedge((FX,FY),ro+1.7,t1+0.3,t2-0.3,width=ro+1.7-R0,facecolor=CAT_COLORS[cat],
                     alpha=0.22,ec="none",zorder=2))
    ax.add_patch(Wedge((FX,FY),ro,t1+0.7,t2-0.7,width=ro-R0,facecolor=CAT_COLORS[cat],
                 alpha=(0.97 if cat=="Neurological" else 0.80),ec="white",lw=0.4,
                 zorder=(4 if cat=="Neurological" else 3)))
for c,st in [(0.5,"dotted"),(0.688,(0,(4,2)))]:
    r=rr(c); th=np.linspace(math.radians(A0),math.radians(A0+SPAN),120)
    ax.plot(FX+r*np.cos(th),FY+r*np.sin(th),color=C_SUB,lw=0.55,ls=st,alpha=0.7,zorder=2)
# mortality thin black anchor wedge below the right end
mr=rr(0.767)*0.62; ma=A0+SPAN+3.5    # left end of the fan
ax.add_patch(Wedge((FX,FY),mr,ma-2.2,ma+2.0,width=mr-R0,facecolor=C_MORT,alpha=0.9,ec="white",lw=0.3,zorder=3))
mtx,mty=FX+mr*math.cos(math.radians(ma)),FY+mr*math.sin(math.radians(ma))
ax.plot([mtx,56],[mty,37.2],color="#aaaaaa",lw=0.5,zorder=2)
T(54,36.2,"mortality",size=4.8,color="#555",ha="center",va="top")
T(54,33.5,"anchor (0.77)",size=4.8,color="#555",ha="center",va="top")
# neuro brain callout
ni=[i for i,(c,_) in enumerate(arr) if c=="Neurological"][0]
nth=math.radians(A0+(ni+0.5)*SPAN/n); nro=rr(0.729)
bx,by=FX+(nro+5.5)*math.cos(nth),FY+(nro+5.5)*math.sin(nth)
icon(ax,"brain",bx,by,8.5,color=C_NEURO,lw=1.7,z=6)
T(FX,FY-4.5,"390 incident outcomes",size=7.0,weight="bold")
T(FX,FY-8.0,"mean concordance 0.688",size=6.4,color="#333")

# PD inset (the only real data): reversed-time td-AUROC
bx0,bx1,by0,by1=58,102,76,95
ax.add_patch(FancyBboxPatch((bx0,by0),bx1-bx0,by1-by0,boxstyle="round,pad=0.4,rounding_size=1.2",
    fc="white",ec="#b9b9b9",lw=0.7,zorder=7))
ax.plot([bx,(bx0+bx1)/2],[by+3,by0-0.3],color="#c4c4c4",lw=0.6,zorder=3)
T((bx0+bx1)/2,by1-1.3,"Parkinson's disease",size=5.8,weight="bold",va="top")
T((bx0+bx1)/2,by1-4.4,"flagged years before diagnosis",size=5.2,va="top",color="#333")
def gx(yr): return bx0+11+(bx1-9-(bx0+11))*((yr+8)/8.0)
def gy(a):  return by0+3.6+9.4*((a-0.5)/0.5)
ax.plot([gx(-8.2),gx(0)],[gy(0.5),gy(0.5)],color=C_SUB,lw=0.5,ls="dotted",zorder=7)
arrow((gx(-8.2),gy(0.5)),(gx(0)+1.4,gy(0.5)),"#999",0.7,z=8)
ax.plot([gx(y) for y in (-7,-5,-3)],[gy(a) for a in (0.87,0.90,0.90)],"-o",
        color=C_NEURO,ms=2.5,lw=1.4,zorder=9)
T(gx(-2.6),gy(0.90),"AUROC ≈ 0.90",size=4.6,color=C_NEURO,ha="left",va="center")
ax.plot([gx(-4.9)]*2,[gy(0.5),gy(0.64)],color="#888",lw=0.6,zorder=8)
T(gx(-4.9),gy(0.66),"median 4.9 y",size=4.5,color="#555",ha="center",va="bottom")
T(gx(0)+1.6,gy(0.5)-0.9,"diagnosis",size=4.7,color="#555",va="top",ha="right")
T(bx0+2.5,by0+1.4,"AD / dementia weaker",size=4.3,color="#999",ha="left",va="bottom")

# ===================== ZONE 4: ORTHOGONAL AXES (secondary hero) =====================
T(131,70,"Genetics adds an independent axis (+0.042)",size=6.2,weight="bold")
T(131,66.5,"for the diseases the wrist reads less well",size=5.9)
OX,OY=112,29; XL,YL=150,54
icon(ax,"dna",116,60.5,6.5,color=C_GEN,lw=1.5)
arrow((OX,OY),(XL,OY),C_WRIST,1.1,z=5)
arrow((OX,OY),(OX,YL),C_GEN,1.1,z=5)
T((OX+XL)/2+3,OY-3,"Wrist signal",size=5.7,color=C_WRIST)
T(OX-2.8,(OY+YL)/2,"Genetics (PRS)",size=5.7,color=C_GEN,rot=90)
ax.plot([OX+2.2,OX+2.2,OX],[OY,OY+2.2,OY+2.2],color="#555",lw=0.7,zorder=6)  # right-angle mark
# neurodegeneration: far out on the wrist axis (genetics adds ~ 0)
arrow((OX,OY),(145,OY+1.3),"#444",1.4,z=6)
ax.add_patch(Circle((145,OY+1.3),1.0,color=C_NEURO,zorder=8))
T(145,OY+3.6,"neurodegeneration",size=4.9,color="#555",ha="center")
T(145,OY-2.0,"wrist only",size=4.7,color="#777",ha="center")
# cardiometabolic / autoimmune: both modalities contribute -> resultant beyond either
cmx,cmy=129,OY+15
arrow((OX,OY),(cmx,OY),C_WRIST,0.9,z=6,ls=(0,(3,2)),head=False)
arrow((cmx,OY),(cmx,cmy),C_GEN,0.9,z=6,ls=(0,(3,2)),head=False)
arrow((OX,OY),(cmx,cmy),"#222",1.7,z=7)
ax.add_patch(Circle((cmx,cmy),1.0,color="#882255",zorder=8))
T(cmx,cmy+3.0,"cardiometabolic /",size=4.9,color="#333",ha="center")
T(cmx,cmy+1.2,"autoimmune",size=4.9,color="#333",ha="center")
rad=math.hypot(cmx-OX,cmy-OY)+2.5; tt=np.linspace(math.radians(2),math.radians(64),50)
ax.plot(OX+rad*np.cos(tt),OY+rad*np.sin(tt),color="#cccccc",lw=0.6,ls="dotted",zorder=3)

# ===================== ZONE 5: PAYOFF =====================
icon(ax,"building-hospital",164,64,9,color=C_SPINE,lw=1.5)
T(164,55,"Earlier treatment",size=6.0,weight="bold")
T(164,51.5,"window",size=6.0,weight="bold")
T(164,47.7,"+ enrich prevention trials",size=5.4)
ax.add_patch(FancyBboxPatch((155,35.5),18,9,boxstyle="round,pad=0.3,rounding_size=1.5",
    fc="#f4f4f4",ec=C_SPINE,lw=0.8,zorder=6))
T(164,40,"12.9×",size=10,weight="bold",color=C_NEURO)
T(164,31.5,"enrichment at\n95% specificity",size=5.3,color="#444",va="top")

_OUT=Path(os.environ.get("UKB_ROOT","data"))/"figures_out"
_OUT.mkdir(parents=True,exist_ok=True)
fig.savefig(str(_OUT/"_fig1_draft.png"),dpi=300,facecolor="white")
fig.savefig(str(_OUT/"_fig1_draft.pdf"),facecolor="white")
print("saved _fig1_draft")
