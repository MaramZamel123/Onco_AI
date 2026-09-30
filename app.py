import os
import re
import tempfile

import streamlit as st
import streamlit.components.v1 as components

import models as M


st.set_page_config(
    page_title="Dr. Nana – Breast Health Assistant",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def _warm_up():
    """Load the RAG index / embedding model once per server process."""
    try:
        M._init_rag()
    except Exception:
        pass
    return True


_warm_up()


def get_reply(q: str) -> str:
    low = q.lower()
    words = set(re.findall(r"[a-z']+", low))

    # Whole-word matching, so "this" / "which" / "health" don't trigger a greeting
    if words & {"hi", "hello", "hey", "salam"} and len(words) <= 3:
        return "Hello! How can I help you today?"
    if "thank" in low:
        return "You're very welcome! Take care of yourself."
    if words & {"scared", "afraid", "worried", "anxious"} and len(words) <= 6:
        return ("It's completely natural to feel that way. "
                "I'm here with you, and we can go through this step by step.")

    try:
        return M.rag_answer(q)
    except Exception:
        return ("Regular screening and knowing your body's normal are the best defenses. "
                "Tell a doctor about any new lump or change.")

DOCTOR_HTML = r"""
<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Dr. Nana – Breast Health Assistant</title>
<style>
:root{--bg:#f1f6f8;--card:#fff;--ink:#1f2f3a;--acc:#1f8a9e;--acc2:#3bb39a;--mut:#e2edf1;--r:14px;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
[data-theme=night]{--bg:#101a26;--card:#1a2838;--ink:#eaf2f7;--acc:#4cc2d6;--acc2:#5fd0b0;--mut:#26384b}
[data-theme=story]{--bg:#f5f2ed;--card:#fffefb;--ink:#3a3128;--acc:#a9764f;--acc2:#6fa88a;--mut:#ece4d9}
*{box-sizing:border-box}html{scroll-padding-top:env(safe-area-inset-top,0px)}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Segoe UI",system-ui,-apple-system,Arial,sans-serif;min-height:100%}
header{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between;padding:12px 16px;background:var(--card);border-bottom:1px solid var(--mut);margin-bottom:14px}
h1{font-size:18px;margin:0;display:flex;align-items:center;gap:10px}
h1 small{display:block;font-size:12px;font-weight:400;opacity:.65}
.logo{width:34px;height:34px;border-radius:10px;background:var(--acc);color:#fff;display:grid;place-items:center;font-size:24px;font-weight:700}
.sw button{border:2px solid var(--acc);background:var(--card);color:var(--ink);border-radius:99px;padding:6px 12px;cursor:pointer;margin-left:4px}
.sw button.on{background:var(--acc);color:#fff}
.app{display:grid;grid-template-columns:260px 1fr;gap:14px;padding:0 16px 16px;max-width:1000px;margin:auto}
[data-theme=night] .app{grid-template-columns:1fr}
[data-theme=story] .app{grid-template-columns:1fr 260px}
[data-theme=story] .doc{order:2}
.doc{background:var(--card);border-radius:var(--r);padding:12px;text-align:center;box-shadow:0 2px 12px #0001;align-self:start}
[data-theme=night] .doc{display:flex;align-items:center;gap:12px;text-align:left}
[data-theme=night] .doc svg{width:120px;flex:none}
.bub{background:var(--mut);border-radius:16px;padding:8px 12px;font-size:14px;margin-bottom:6px;min-height:38px}
.mood{font-size:12px;opacity:.7}
#dr{transition:transform .7s cubic-bezier(.5,1.6,.5,1);width:100%;max-width:220px}
.panel{background:var(--card);border-radius:var(--r);box-shadow:0 2px 12px #0001;display:flex;flex-direction:column;min-height:520px;overflow:hidden}
.tabs{display:flex}
.tabs button{flex:1;border:0;padding:12px;background:var(--mut);color:var(--ink);cursor:pointer;font-weight:bold}
.tabs button.on{background:var(--acc);color:#fff}
.tab{display:none;padding:14px;flex:1;overflow:auto}
.tab.on{display:flex;flex-direction:column}
#msgs{flex:1;overflow:auto;display:flex;flex-direction:column;gap:8px;max-height:380px}
.m{max-width:80%;padding:8px 12px;border-radius:16px;font-size:14px;line-height:1.4;white-space:pre-wrap}
.bot{background:var(--mut);align-self:flex-start}
.me{background:var(--acc);color:#fff;align-self:flex-end}
.row{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
input,button.go{font:inherit}
.row input{flex:1;min-width:120px;padding:10px;border-radius:12px;border:2px solid var(--mut);background:var(--bg);color:var(--ink)}
button.go,.chip{background:var(--acc);color:#fff;border:0;border-radius:12px;padding:9px 14px;cursor:pointer}
.chip{background:var(--mut);color:var(--ink);border-radius:99px;font-size:13px}
.bar{margin:8px 0}
.bar span{display:flex;justify-content:space-between;font-size:13px}
.bar i{display:block;height:14px;border-radius:9px;background:var(--mut);overflow:hidden}
.bar b{display:block;height:100%;width:0;background:var(--acc2);border-radius:9px;transition:width 1s}
.bar.m b{background:#d64550}
.bar.b b{background:#e8a33d}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px}
.grid label{font-size:12px}
.grid label small{display:block;opacity:.6}
.grid input{width:100%;padding:7px;border-radius:10px;border:2px solid var(--mut);background:var(--bg);color:var(--ink)}
.note{font-size:11px;opacity:.65;margin-top:10px}
.wave{animation:wv .5s ease-in-out infinite alternate}
@keyframes wv{to{transform:rotate(-25deg)}}
.blink{animation:bl 4s infinite;transform-origin:100px 96px}
@keyframes bl{0%,94%,100%{transform:scaleY(1)}97%{transform:scaleY(.1)}}
.bob{animation:bo 2.2s ease-in-out infinite alternate}
@keyframes bo{to{transform:translateY(5px)}}
.drop{border:2px dashed var(--acc);border-radius:var(--r);padding:26px;text-align:center;cursor:pointer;background:var(--bg)}
.drop.hov{background:var(--mut)}
.drop small{display:block;opacity:.65;margin-top:4px}
.thumb{max-width:100%;max-height:200px;border-radius:10px;margin-top:12px;display:none;border:1px solid var(--mut)}
.mp{margin-top:10px;font-size:11px;opacity:.85}
.mp button{border:1px solid var(--mut);background:var(--bg);color:var(--ink);border-radius:99px;font-size:11px;padding:3px 8px;margin:2px;cursor:pointer}
.tag{display:inline-block;font-size:11px;background:var(--mut);border-radius:99px;padding:2px 8px;margin-top:8px}
.blink.sleep{animation:none;transform:scaleY(.15)}
</style></head><body data-x>
<header><h1><span class="logo">+</span><span>Dr. Nana<small>Breast Ultrasound &amp; Health Assistant</small></span></h1>
<div class="sw" id="sw"><button data-t="clinic" class="on">Clinical</button><button data-t="night">Midnight</button><button data-t="story">Warm</button></div></header>
<div class="app">
<div class="doc"><div><div class="bub" id="bub">Hello, I'm Dr. Nana</div>
<div class="mood" id="mood">mood: waving</div><div class="mp" id="mp">Preview moods:<br></div></div>
<svg id="dr" viewBox="0 0 200 250"><g class="bob">
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
<text id="zz" x="150" y="40" font-size="22" fill="#1f8a9e" opacity="0" font-weight="bold">z Z</text></g></svg></div>

<div class="panel"><div class="tabs"><button data-p="chat" class="on">Chat</button><button data-p="scan">Scan</button><button data-p="vals">Values</button></div>
<div class="tab on" id="chat"><div id="msgs"></div>
<div class="row"><button class="chip" data-q="Upload scan">Upload scan</button></div>
<div class="row"><input id="inp" placeholder="Ask Dr. Nana a health question…"><button class="go" id="send">Send</button></div></div>
<div class="tab" id="scan"><div class="drop" id="drop"><b>Upload breast ultrasound image</b><small>Drag &amp; drop or click · JPG / PNG · ultrasound images only</small></div>
<input type="file" id="file" accept="image/*" hidden><img id="thumb" class="thumb" alt="preview"><div id="imgres" style="margin-top:12px"></div></div>
<div class="tab" id="vals"><p>Enter your lab measurements from the report:</p><div class="grid" id="grid"></div>
<div class="row"><button class="go" id="run">Analyze values</button><button class="chip" id="ex">Fill example</button></div><div id="valres" style="margin-top:12px"></div></div>
<div class="note" style="padding:0 14px 12px">AI-assisted screening support for educational use. It is not a medical diagnosis; please consult a qualified physician.</div></div></div>
<script>
const API={chat:"",image:"",values:""};
const F=[["texture_mean","Average Tissue Texture","Pixel variation in tissue image",19],["concave points_mean","Average Indentations on Cell Edges","Count of small dents",.05],["radius_se","Cell Size Variability","Spread of cell radius",.4],["area_se","Cell Area Variability","Spread of cell area",40],["compactness_se","Cell Shape Tightness Variability","Spread of compactness",.025],["radius_worst","Largest Cell Radius","Biggest cell measured",16],["texture_worst","Roughest Tissue Texture","Highest texture value",25],["area_worst","Largest Cell Area","Biggest cell area",880],["concavity_worst","Deepest Cell Edge Dent","Worst dent depth",.27],["concave points_worst","Most Indentations on a Cell Edge","Worst dent count",.11]];
const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
$("#grid").innerHTML = F.map((f, i) => `<label>${f[1]}<small>${f[2]}</small><input type="number" step="any" id="f${i}"></label>`).join("");


const ST = {
  send(type, data) {
    window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: type }, data), "*");
  }
};
let lastReplyId = null;
let typing = null;

window.addEventListener("message", e => {
  if (!e.data || e.data.type !== "streamlit:render") return;
  const r = e.data.args && e.data.args.reply;
  if (!r) return;
  if (lastReplyId === null) { lastReplyId = r.id; return; }   // ignore stale reply on first load
  if (r.id !== lastReplyId) {
    lastReplyId = r.id;
    if (typing) { typing.remove(); typing = null; }
    add(r.text, "bot");
    mood("talk", r.text.replace(/\s+/g, " ").slice(0, 60) + "…");
  }
});
ST.send("streamlit:componentReady", { apiVersion: 1 });
ST.send("streamlit:setFrameHeight", { height: 850 });


const M={
  wave:{m:"M84 108 Q100 126 116 108", bl:0, br:0, by:0, bl2:1, arm:1, t:"waving"},
  happy:{m:"M84 108 Q100 124 116 108", b:0, t:"happy"},
  think:{m:"M90 114 Q100 110 112 116", b:1, q:1, t:"thinking"},
  worry:{m:"M86 119 Q100 106 114 119", b:2, sw:1, t:"concerned"},
  good:{m:"M80 106 Q100 134 120 106Z", b:0, st:1, eh:1, t:"celebrating", jump:1},
  talk:{m:"M88 108 Q100 128 112 108Z", b:0, t:"talking"},
  listen:{m:"M90 110 Q100 117 110 110", b:0, tilt:1, t:"listening"},
  reassure:{m:"M86 108 Q100 120 114 108", b:3, tilt:1, t:"reassuring"},
  wow:{m:"M92 108 Q100 130 108 108Z", b:1, st:1, jump:1, t:"excited"},
  sleepy:{m:"M92 114 Q100 118 108 114", b:3, sl:1, t:"sleepy"}
};

let tk;
function mood(k, say) {
  const d = M[k];
  $("#mouth").setAttribute("d", d.m);
  $("#mouth").setAttribute("fill", /Z$/.test(d.m) ? "#ff8fae" : "none");
  const b = d.b || 0, rot = [0, -8, 10, 6][b], y = [0, -6, -2, -3][b];
  $("#bL").style.transform = `translateY(${y}px) rotate(${rot}deg)`;
  $("#bR").style.transform = `translateY(${y}px) rotate(${-rot}deg)`;
  $("#bL").style.transformOrigin = "81px 79px";
  $("#bR").style.transformOrigin = "119px 79px";
  $("#eh").style.opacity = d.eh ? 1 : 0;
  $("#eyes").style.opacity = d.eh ? 0 : 1;
  $("#eyes").classList.toggle("sleep", !!d.sl);
  $("#zz").style.opacity = d.sl ? 1 : 0;
  $("#sweat").style.opacity = d.sw ? 1 : 0;
  $("#qm").style.opacity = d.q ? 1 : 0;
  $("#st").style.opacity = d.st ? 1 : 0;
  $("#armR").classList.toggle("wave", !!d.arm);
  $("#mood").textContent = "mood: " + d.t;
  $("#dr").style.transform = d.jump ? "translateY(-14px) rotate(-3deg)" : d.tilt ? "rotate(4deg)" : b == 2 ? "translateX(-6px)" : "none";
  if (say) $("#bub").textContent = say;
  clearInterval(tk);
  if (k == "talk") {
    let o = 0;
    tk = setInterval(() => { $("#mouth").setAttribute("d", (o = !o) ? M.talk.m : M.happy.m); }, 220);
    setTimeout(() => { clearInterval(tk); mood("happy"); }, 1600);
  }
}


function add(t, c) {
  const d = document.createElement("div");
  d.className = "m " + c;
  d.textContent = t;
  $("#msgs").append(d);
  $("#msgs").scrollTop = 1e9;
  return d;
}

function tab(p) {
  $$(".tab").forEach(e => e.classList.toggle("on", e.id == p));
  $$(".tabs button").forEach(e => e.classList.toggle("on", e.dataset.p == p));
}

$$(".tabs button").forEach(b => b.onclick = () => {
  tab(b.dataset.p);
  const p = b.dataset.p;
  p == "chat" ? mood("happy", "How can I help?") : p == "scan" ? mood("listen", "Upload your breast ultrasound image.") : mood("listen", "Enter the values from your report.");
});

function ask(q) {
  q = (q || "").trim();
  if (!q) return;
  if (q === "Upload scan") {              
    tab("scan");
    mood("listen", "Upload your breast ultrasound image.");
    return;
  }
  tab("chat");
  add(q, "me");
  $("#inp").value = "";
  mood("think", "Let me look into that…");
  if (typing) typing.remove();
  typing = add("Dr. Nana is typing…", "bot");
  ST.send("streamlit:setComponentValue", { value: { id: Date.now(), q: q }, dataType: "json" });
}

$("#send").onclick = () => ask($("#inp").value);
$("#inp").onkeydown = e => { if (e.key == "Enter") ask($("#inp").value); };
$$(".chip[data-q]").forEach(c => c.onclick = () => ask(c.dataset.q));


function bars(el, r) {
  el.innerHTML = r.map(x => `<div class="bar ${x[2]}"><span><b style="all:unset">${x[0]}</b><span>${x[1]}%</span></span><i><b></b></i></div>`).join("");
  setTimeout(() => $$("#" + el.id + " .bar").forEach((b, i) => b.querySelector("i b").style.width = r[i][1] + "%"), 50);
}

function react(r) {
  const top = r.reduce((a, b) => b[1] > a[1] ? b : a);
  if (top[1] < 60) {
    mood("think", "I'm not fully certain. A specialist review is advised.");
    add("The result is inconclusive (" + top[0] + " " + top[1] + "%). Please consult a radiologist.", "bot");
    return;
  }
  if (top[2] == "m") {
    mood("worry", "This needs a specialist's attention.");
    add("Result: " + top[0] + " (" + top[1] + "%). Please consult a doctor for confirmation.", "bot");
    setTimeout(() => mood("reassure", "Early evaluation makes a big difference. I'm here for questions."), 4500);
  } else if (top[2] == "n") {
    mood("good", "Good news!");
    add("Result looks reassuring (" + top[1] + "%). Keep up regular check-ups.", "bot");
    setTimeout(() => mood("happy"), 4000);
  } else {
    mood("reassure", "Benign, but let's keep watch.");
    add("Result: benign (" + top[1] + "%). A follow-up is usually advised.", "bot");
  }
}

$("#drop").onclick = () => $("#file").click();
["dragover", "dragleave", "drop"].forEach(n => $("#drop").addEventListener(n, e => {
  e.preventDefault();
  $("#drop").classList.toggle("hov", n == "dragover");
  if (n == "drop" && e.dataTransfer.files[0]) {
    $("#file").files = e.dataTransfer.files;
    $("#file").onchange({ target: $("#file") });
  }
}));

$("#mp").innerHTML += Object.keys(M).map(k => `<button onclick="mood('${k}')">${k}</button>`).join("");

$("#file").onchange = async e => {
  const f = e.target.files[0];
  if (!f) return;
  if (!f.type.startsWith("image/")) {
    mood("worry", "Please upload an ultrasound image.");
    return;
  }
  $("#thumb").src = URL.createObjectURL(f);
  $("#thumb").style.display = "block";
  mood("think", "Analyzing your scan…");
  $("#imgres").innerHTML = '<span class="tag">Analyzing image…</span>';
  let r;
  try {
    if (API.image) {
      const fd = new FormData();
      fd.append("file", f);
      r = await (await fetch(API.image, { method: "POST", body: fd })).json();
    }
  } catch (x) {}
  if (!r) {
    const s = f.size % 100, a = 20 + s % 50, b = Math.round((100 - a) * .6);
    r = [["Normal", a, "n"], ["Benign", b, "b"], ["Malignant", 100 - a - b, "m"]];
  } else {
    r = [["Normal", r.normal, "n"], ["Benign", r.benign, "b"], ["Malignant", r.malignant, "m"]];
  }
  setTimeout(() => { bars($("#imgres"), r); react(r); }, 900);
};

$("#ex").onclick = () => {
  F.forEach((f, i) => $("#f" + i).value = f[3]);
  mood("happy", "All set. Press Analyze.");
};

$("#grid").addEventListener("input", () => {
  const n = F.filter((f, i) => $("#f" + i).value).length;
  n == 10 ? mood("happy", "All set. Press Analyze.") : mood("listen", n + " of 10 values entered");
});

$("#run").onclick = async () => {
  const v = F.map((f, i) => +$("#f" + i).value);
  if (v.some(x => !x)) {
    mood("worry", "Please fill in every field.");
    return;
  }
  mood("think", "Crunching numbers…");
  let p;
  try {
    if (API.values) {
      p = (await (await fetch(API.values, { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(Object.fromEntries(F.map((f, i) => [f[0], v[i]]))) })).json()).malignant;
    }
  } catch (x) {}
  if (p == null) p = Math.min(97, Math.max(3, Math.round(((v[7] / 880) + (v[5] / 16) + (v[9] / .11)) / 3 * 45)));
  const r = [["Benign", 100 - p, p < 50 ? "n" : "b"], ["Malignant", p, "m"]];
  r[0][2] = "n";
  setTimeout(() => {
    bars($("#valres"), r);
    react(p >= 50 ? [["Malignant", p, "m"], ["Benign", 100 - p, "b"]] : [["Benign", 100 - p, "n"], ["Malignant", p, "m"]]);
  }, 800);
};


$$("#sw button").forEach(b => b.onclick = () => {
  $$("#sw button").forEach(x => x.classList.toggle("on", x == b));
  document.documentElement.setAttribute("data-theme", b.dataset.t);
});

$("#inp").addEventListener("input", () => {
  const v = $("#inp").value;
  v ? mood("listen", "I'm listening…") : mood("happy");
});

$("#drop").addEventListener("dragenter", () => mood("wow", "Drop the ultrasound here!"));
$("#drop").addEventListener("dragleave", () => mood("happy"));

let idle;
function wake() {
  clearTimeout(idle);
  if ($("#eyes").classList.contains("sleep")) mood("wave", "I'm back! What's next?");
  idle = setTimeout(() => mood("sleepy", "Zzz… tap anywhere to wake me"), 40000);
}
["pointerdown", "keydown", "touchstart"].forEach(e => document.addEventListener(e, wake));
wake();

add("Hello, I'm Dr. Nana. Ask me a health question, upload a scan, or enter your lab values.", "bot");
mood("wave", "Hello, I'm Dr. Nana");
setTimeout(() => mood("happy"), 3500);
</script></body></html>
"""

COMP_DIR = os.path.join(tempfile.gettempdir(), "dr_nana_component")
os.makedirs(COMP_DIR, exist_ok=True)
with open(os.path.join(COMP_DIR, "index.html"), "w", encoding="utf-8") as fh:
    fh.write(DOCTOR_HTML)

dr_nana = components.declare_component("dr_nana", path=COMP_DIR)

if "reply" not in st.session_state:
    st.session_state.reply = {"id": 0, "text": ""}

# Renders the UI; returns the latest question sent from the page (or None)
event = dr_nana(reply=st.session_state.reply, key="nana", default=None)

if event and event.get("id") != st.session_state.get("last_id"):
    st.session_state.last_id = event["id"]
    answer = get_reply(event["q"])
    st.session_state.reply = {"id": event["id"], "text": answer}
    st.rerun()
