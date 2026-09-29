"""Dr. Nana avatar for Streamlit: renders the animated SVG with a mood set from Python."""
import json
import streamlit.components.v1 as components

SVG = r"""<svg id="dr" viewBox="0 0 200 250"><g class="bob">
<path d="M46 96 Q40 16 100 16 Q160 16 154 96 L168 190 Q100 212 32 190Z" fill="#17151b"/>
<path d="M36 250 Q38 168 74 150 L126 150 Q162 168 164 250Z" fill="#8a8563"/>
<circle cx="120" cy="196" r="3.5" fill="#d8b45a"/><circle cx="122" cy="226" r="3.5" fill="#d8b45a"/>
<g id="armL"><path d="M50 162 Q26 200 40 228" stroke="#8a8563" stroke-width="22" stroke-linecap="round" fill="none"/><circle cx="40" cy="232" r="10" fill="#f3c6a5"/></g>
<g id="armR" style="transform-origin:150px 162px"><path d="M150 162 Q176 158 178 124" stroke="#8a8563" stroke-width="22" stroke-linecap="round" fill="none"/><circle cx="178" cy="114" r="10" fill="#f3c6a5"/></g>
<path d="M66 138 Q100 170 134 138 L142 222 Q100 240 58 222Z" fill="#17151b"/>
<path d="M84 150 Q100 176 116 150" stroke="#6fa3b0" stroke-width="3" fill="none"/><path d="M84 150 Q78 190 100 196 Q114 194 116 182" stroke="#6fa3b0" stroke-width="3" fill="none"/><circle cx="116" cy="182" r="5" fill="#9fbfc8"/>
<rect x="44" y="196" width="26" height="14" rx="3" fill="#fff"/><rect x="47" y="199" width="12" height="2.5" fill="#1f8a9e"/><rect x="47" y="204" width="18" height="2" fill="#9ab"/>
<ellipse cx="100" cy="94" rx="44" ry="46" fill="#f3c6a5"/>
<path fill-rule="evenodd" fill="#17151b" d="M46 96 Q40 16 100 16 Q160 16 154 96 Q152 146 100 156 Q48 146 46 96Z M100 48 Q144 48 144 94 Q142 138 100 142 Q58 138 56 94 Q56 48 100 48Z"/>
<circle cx="70" cy="112" r="8" fill="#f4a3b0" id="blL" opacity=".5"/><circle cx="130" cy="112" r="8" fill="#f4a3b0" id="blR" opacity=".5"/>
<g class="blink" id="eyes"><ellipse cx="82" cy="96" rx="8.5" ry="10.5" fill="#3a2418"/><ellipse cx="118" cy="96" rx="8.5" ry="10.5" fill="#3a2418"/>
<circle cx="85" cy="92" r="3.4" fill="#fff"/><circle cx="121" cy="92" r="3.4" fill="#fff"/><circle cx="79" cy="100" r="1.6" fill="#fff"/><circle cx="115" cy="100" r="1.6" fill="#fff"/></g>
<path id="eh" d="M72 98 Q82 86 92 98 M108 98 Q118 86 128 98" stroke="#3a2418" stroke-width="3.5" fill="none" stroke-linecap="round" opacity="0"/>
<path id="bL" d="M71 80 Q81 75 91 79" stroke="#2a1e18" stroke-width="3" fill="none" stroke-linecap="round"/>
<path id="bR" d="M109 79 Q119 75 129 80" stroke="#2a1e18" stroke-width="3" fill="none" stroke-linecap="round"/>
<path d="M98 108 Q100 111 102 108" stroke="#d9a07f" stroke-width="2" fill="none" stroke-linecap="round"/>
<path id="mouth" style="transform:translateY(8px)" d="M84 108 Q100 124 116 108" stroke="#c0394f" stroke-width="3.5" fill="none" stroke-linecap="round"/>
<path id="sweat" d="M146 84 q6 10 0 14 q-6 -4 0 -14" fill="#8fd3ff" opacity="0"/>
<text id="qm" x="152" y="34" font-size="26" fill="#1f8a9e" opacity="0" font-weight="bold">?</text>
<text id="st" x="14" y="50" font-size="24" opacity="0">✨</text>
<text id="zz" x="150" y="40" font-size="22" fill="#1f8a9e" opacity="0" font-weight="bold">z Z</text></g></svg>"""

MOOD_JS = r"""const M={wave:{m:"M84 108 Q100 126 116 108",bl:0,br:0,by:0,bl2:1,arm:1,t:"waving"},happy:{m:"M84 108 Q100 124 116 108",b:0,t:"happy"},
think:{m:"M90 114 Q100 110 112 116",b:1,q:1,t:"thinking"},worry:{m:"M86 119 Q100 106 114 119",b:2,sw:1,t:"concerned"},
good:{m:"M80 106 Q100 134 120 106Z",b:0,st:1,eh:1,t:"celebrating",jump:1},talk:{m:"M88 108 Q100 128 112 108Z",b:0,t:"talking"},
listen:{m:"M90 110 Q100 117 110 110",b:0,tilt:1,t:"listening"},reassure:{m:"M86 108 Q100 120 114 108",b:3,tilt:1,t:"reassuring"},
wow:{m:"M92 108 Q100 130 108 108Z",b:1,st:1,jump:1,t:"excited"},sleepy:{m:"M92 114 Q100 118 108 114",b:3,sl:1,t:"sleepy"}};
let tk;function mood(k,say){const d=M[k];$("#mouth").setAttribute("d",d.m);$("#mouth").setAttribute("fill",/Z$/.test(d.m)?"#ff8fae":"none");
const b=d.b||0,rot=[0,-8,10,6][b],y=[0,-6,-2,-3][b];$("#bL").style.transform=`translateY(${y}px) rotate(${rot}deg)`;$("#bR").style.transform=`translateY(${y}px) rotate(${-rot}deg)`;
$("#bL").style.transformOrigin="81px 79px";$("#bR").style.transformOrigin="119px 79px";$("#eh").style.opacity=d.eh?1:0;$("#eyes").style.opacity=d.eh?0:1;$("#eyes").classList.toggle("sleep",!!d.sl);$("#zz").style.opacity=d.sl?1:0;
$("#sweat").style.opacity=d.sw?1:0;$("#qm").style.opacity=d.q?1:0;$("#st").style.opacity=d.st?1:0;
$("#armR").classList.toggle("wave",!!d.arm);$("#mood").textContent="mood: "+d.t;
$("#dr").style.transform=d.jump?"translateY(-14px) rotate(-3deg)":d.tilt?"rotate(4deg)":b==2?"translateX(-6px)":"none";
if(say)$("#bub").textContent=say;clearInterval(tk);if(k=="talk"){let o=0;tk=setInterval(()=>{$("#mouth").setAttribute("d",(o=!o)?M.talk.m:M.happy.m)},220);setTimeout(()=>{clearInterval(tk);mood("happy")},1600)}}
"""

PAGE = r"""<style>
body{margin:0;font-family:"Segoe UI",system-ui,sans-serif;text-align:center;background:transparent}
#bub{background:#e2edf1;color:#1f2f3a;border-radius:14px;padding:8px 12px;font-size:14px;margin:4px 8px 6px;min-height:36px}
.mood{font-size:12px;color:#1f2f3a;opacity:.6}#dr{max-width:230px;width:100%;transition:transform .7s cubic-bezier(.5,1.6,.5,1)}
.wave{animation:wv .5s ease-in-out infinite alternate}@keyframes wv{to{transform:rotate(-25deg)}}
.blink{animation:bl 4s infinite;transform-origin:100px 96px}@keyframes bl{0%,94%,100%{transform:scaleY(1)}97%{transform:scaleY(.1)}}
.bob{animation:bo 2.2s ease-in-out infinite alternate}@keyframes bo{to{transform:translateY(5px)}}
.blink.sleep{animation:none;transform:scaleY(.15)}
</style>
<div id="bub"></div><div class="mood" id="mood"></div>__SVG__
<!--nonce __N__-->
<script>const $=s=>document.querySelector(s);
__JS__
mood(__MOOD__,__SAY__);
if(["wave","good"].includes(__MOOD__))setTimeout(()=>mood("happy"),4000);
</script>"""


def render_doctor(mood: str, say: str, nonce: int = 0, height: int = 420):
    html = (PAGE.replace("__SVG__", SVG).replace("__JS__", MOOD_JS)
            .replace("__MOOD__", json.dumps(mood)).replace("__SAY__", json.dumps(say))
            .replace("__N__", str(nonce)))
    components.html(html, height=height)
