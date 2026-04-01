"""Plotly legend stats — client-side regression that updates on legend clicks.

Uses jStat (CDN) for the t-distribution p-value, with an inline
regularised-incomplete-beta fallback if the CDN is blocked.
All other stats (R², slope, intercept) are plain arithmetic.
No Python round-trip needed — stats update instantly on legend toggle.
"""

import json
import streamlit as st
from typing import Optional, Any


# ── HTML + JS component ─────────────────────────────────────────────
# Renders a stats bar inside the iframe and attaches a plotly_restyle
# listener to the chart identified by fig.layout.meta in the parent.
_COMPONENT_HTML = r"""<!DOCTYPE html>
<html><head>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:"Source Sans Pro",sans-serif;background:transparent}
  .row{display:flex;gap:0;padding:0 4px}
  .m{flex:1}
  .lb{font-size:.92rem;color:__LABEL__;letter-spacing:.01em}
  .vl{font-size:1.45rem;font-weight:700;color:__VALUE__;line-height:1.3}
  .eq{font-size:.82rem;color:__LABEL__;padding:2px 4px 0}
  .eq b{color:__VALUE__;font-weight:600}
  .ht{font-size:.65rem;color:__LABEL__;padding:2px 4px 0}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jstat/1.9.4/jstat.min.js"
        onerror="this.onerror=null;var s=document.createElement('script');
                 s.src='https://cdn.jsdelivr.net/npm/jstat@1.9.6/dist/jstat.min.js';
                 document.head.appendChild(s);"></script>
</head><body>
<div class="row">
  <div class="m"><div class="lb">R&sup2;</div><div class="vl" id="v_r2">&mdash;</div></div>
  <div class="m"><div class="lb">Slope</div><div class="vl" id="v_sl">&mdash;</div></div>
  <div class="m"><div class="lb">Intercept</div><div class="vl" id="v_in">&mdash;</div></div>
  <div class="m"><div class="lb">p-value</div><div class="vl" id="v_p">&mdash;</div></div>
</div>
<div class="eq" id="v_eq"></div>
<div class="ht" id="ht"></div>

<script>
(function(){
  "use strict";
  var META="__CHART_META__",YCOL="__Y_COL__",XCOL="__X_COL__",POLL=200,MAXP=30,DEB=300;
  var attached=null,dbt=null;

  /* ── inline t-distribution fallback (Abramowitz & Stegun) ──── */
  function _lng(x){
    var c=[76.18009172947146,-86.50532032941677,24.01409824083091,
           -1.231739572450155,1.208650973866179e-3,-5.395239384953e-6];
    var y=x,t=x+5.5; t-=(x+0.5)*Math.log(t);
    var s=1.000000000190015;
    for(var j=0;j<6;j++) s+=c[j]/++y;
    return -t+Math.log(2.5066282746310005*s/x);
  }
  function _bcf(x,a,b){
    var qa=a+b,qp=a+1,qm=a-1,c=1,d=1-qa*x/qp;
    if(Math.abs(d)<1e-30) d=1e-30; d=1/d; var h=d;
    for(var m=1;m<=200;m++){
      var m2=2*m,aa=m*(b-m)*x/((qm+m2)*(a+m2));
      d=1+aa*d; if(Math.abs(d)<1e-30) d=1e-30;
      c=1+aa/c; if(Math.abs(c)<1e-30) c=1e-30;
      d=1/d; h*=d*c;
      aa=-(a+m)*(qa+m)*x/((a+m2)*(qp+m2));
      d=1+aa*d; if(Math.abs(d)<1e-30) d=1e-30;
      c=1+aa/c; if(Math.abs(c)<1e-30) c=1e-30;
      d=1/d; h*=d*c;
      if(Math.abs(d*c-1)<3e-7) break;
    }
    return h;
  }
  function _bi(x,a,b){
    if(x<=0) return 0; if(x>=1) return 1;
    var bt=Math.exp(_lng(a+b)-_lng(a)-_lng(b)+a*Math.log(x)+b*Math.log(1-x));
    if(x<(a+1)/(a+b+2)) return bt*_bcf(x,a,b)/a;
    return 1-bt*_bcf(1-x,b,a)/b;
  }
  /* Normal tail probability via log-space (handles extreme t-values) */
  function _normTail2(z){
    /* Two-tailed p from normal approx: 2*phi(z)/z*(1-1/z^2+3/z^4)  */
    var z2=z*z,lp=-0.5*z2-Math.log(z)-0.5*Math.log(2*Math.PI);
    lp+=Math.log(Math.abs(1-1/z2+3/(z2*z2)));
    return 2*Math.exp(lp);
  }
  function pFromT(t,df){
    var at=Math.abs(t),p;
    /* Try jStat */
    if(typeof jStat!=="undefined"){
      try{p=2*(1-jStat.studentt.cdf(at,df));if(p>0)return p;}catch(e){}
    }
    /* Try incomplete beta */
    var x=df/(df+t*t); p=_bi(x,df/2,0.5);
    if(p>0) return p;
    /* Fallback: normal approx in log-space for extreme t-values */
    if(at>1) return _normTail2(at);
    return 0;
  }

  /* ── linear regression ─────────────────────────────────────── */
  function regress(xs,ys){
    var n=xs.length; if(n<2) return null;
    var sx=0,sy=0,sxy=0,sx2=0;
    for(var i=0;i<n;i++){sx+=xs[i];sy+=ys[i];sxy+=xs[i]*ys[i];sx2+=xs[i]*xs[i];}
    var den=n*sx2-sx*sx;
    if(Math.abs(den)<1e-15) return null;
    var sl=(n*sxy-sx*sy)/den, ic=(sy-sl*sx)/n, ym=sy/n;
    var sst=0,ssr=0;
    for(var i=0;i<n;i++){sst+=(ys[i]-ym)*(ys[i]-ym);var r=ys[i]-(sl*xs[i]+ic);ssr+=r*r;}
    var r2=sst>0?1-ssr/sst:0, p=null;
    if(n>2){
      var mse=ssr/(n-2),xm=sx/n,sxx=0;
      for(var i=0;i<n;i++) sxx+=(xs[i]-xm)*(xs[i]-xm);
      if(sxx>0&&mse>=0){var se=Math.sqrt(mse/sxx);if(se>0) p=pFromT(Math.abs(sl/se),n-2);}
    }
    return {r2:r2,slope:sl,intercept:ic,p:p,n:n};
  }

  /* ── gather x,y from visible Plotly traces ─────────────────── */
  function gather(chart){
    var xs=[],ys=[];
    if(!chart._fullData) return {xs:xs,ys:ys};
    for(var i=0;i<chart._fullData.length;i++){
      var t=chart._fullData[i];
      if(t.showlegend===false&&t.mode==="lines") continue;
      if(t.visible==="legendonly"||t.visible===false) continue;
      if(!t.x||!t.y) continue;
      for(var j=0;j<t.x.length;j++){
        var xv=t.x[j],yv=t.y[j];
        if(typeof xv==="number"&&typeof yv==="number"&&isFinite(xv)&&isFinite(yv)){
          xs.push(xv);ys.push(yv);
        }
      }
    }
    return {xs:xs,ys:ys};
  }

  /* ── update DOM ────────────────────────────────────────────── */
  function show(s){
    var r2=document.getElementById("v_r2"),sl=document.getElementById("v_sl"),
        ic=document.getElementById("v_in"),pv=document.getElementById("v_p"),
        eq=document.getElementById("v_eq"),ht=document.getElementById("ht");
    if(!s){r2.textContent=sl.textContent=ic.textContent=pv.textContent="\u2014";
      eq.innerHTML="";ht.textContent="Need \u2265 2 visible data points";return;}
    r2.textContent=s.r2.toFixed(4);
    sl.textContent=s.slope.toFixed(4);
    ic.textContent=s.intercept.toFixed(4);
    pv.textContent=s.p!==null?(s.p===0?"< 1e-300":s.p<0.001?s.p.toExponential(2):s.p.toFixed(4)):"N/A";
    var sign=s.intercept>=0?" + ":" \u2212 ";
    eq.innerHTML="<b>"+YCOL+" = "+s.slope.toFixed(4)+" \u00d7 "+XCOL+sign+Math.abs(s.intercept).toFixed(4)+"</b> &nbsp; (n = "+s.n+")";
    ht.textContent="";
  }

  /* ── pre-fill from Python (avoids flash of empty) ──────────── */
  var INIT=__INITIAL__;
  if(INIT) show(INIT);

  /* ── find chart by meta (multi-strategy) ──────────────────── */
  function _metaMatch(el){
    try{return el._fullLayout&&el._fullLayout.meta===META;}catch(e){return false;}
  }
  function findChart(){
    var doc=window.parent.document,i,j,pd;
    /* Strategy 1: direct search in parent DOM */
    var divs=doc.querySelectorAll(".js-plotly-plot");
    for(i=0;i<divs.length;i++) if(_metaMatch(divs[i])) return divs[i];
    /* Strategy 2: inside Streamlit Plotly containers */
    var containers=doc.querySelectorAll('[data-testid="stPlotlyChart"]');
    for(i=0;i<containers.length;i++){
      pd=containers[i].querySelector(".js-plotly-plot");
      if(pd&&_metaMatch(pd)) return pd;
    }
    /* Strategy 3: inside same-origin iframes */
    var iframes=doc.querySelectorAll("iframe");
    for(i=0;i<iframes.length;i++){
      try{
        var idoc=iframes[i].contentDocument||iframes[i].contentWindow.document;
        var idivs=idoc.querySelectorAll(".js-plotly-plot");
        for(j=0;j<idivs.length;j++) if(_metaMatch(idivs[j])) return idivs[j];
      }catch(e){}
    }
    return null;
  }

  function refresh(ch){var d=gather(ch);show(regress(d.xs,d.ys));}
  function restyleHandler(){
    clearTimeout(dbt);
    dbt=setTimeout(function(){if(attached) refresh(attached);},DEB);
  }

  /* ── poll, attach, then watch for chart replacement ──────── */
  var polls=0;
  var tm=setInterval(function(){
    polls++;
    var ch=findChart();
    if(ch){
      clearInterval(tm);
      if(!INIT) refresh(ch);  /* trust Python values on initial load */
      attached=ch;
      ch.on("plotly_restyle",restyleHandler);
      /* Watchdog: re-check every 2s for chart DOM replacement */
      setInterval(function(){
        var ch2=findChart();
        if(ch2&&ch2!==attached){
          attached=ch2;
          refresh(ch2);
          ch2.on("plotly_restyle",restyleHandler);
        }
      },2000);
    }else if(polls>=MAXP){
      clearInterval(tm);
      document.getElementById("ht").textContent="Chart not found";
    }
  },POLL);
})();
</script></body></html>"""


def plotly_legend_monitor(
    chart_meta: str,
    key: str,
    initial_stats: Optional[dict[str, Any]] = None,
    x_col: str = "x",
    y_col: str = "y",
) -> None:
    """Render an interactive stats bar that updates on Plotly legend clicks.

    Place AFTER ``st.plotly_chart()``.  The component finds the chart in
    the parent DOM via ``fig.layout.meta``, reads visible trace data, and
    computes regression statistics entirely client-side (jStat for
    p-values, plain arithmetic for the rest).

    Args:
        chart_meta: Value from ``fig.update_layout(meta=...)``.
        key: Unique Streamlit key for this component instance.
        initial_stats: Pre-computed stats dict with keys
            ``r2``, ``slope``, ``intercept``, ``p``, ``n``.
            Displayed immediately while JS initializes.
        x_col: X-axis column name (for equation display).
        y_col: Y-axis column name (for equation display).
    """
    try:
        is_dark = st.context.theme.base == "dark"
    except Exception:
        is_dark = False

    if is_dark:
        subs = {"__LABEL__": "#999", "__VALUE__": "#fafafa"}
    else:
        subs = {"__LABEL__": "#808495", "__VALUE__": "#0e1117"}

    html = _COMPONENT_HTML
    for placeholder, value in subs.items():
        html = html.replace(placeholder, value)
    html = html.replace("__CHART_META__", chart_meta)
    html = html.replace("__X_COL__", x_col)
    html = html.replace("__Y_COL__", y_col)
    html = html.replace(
        "__INITIAL__",
        json.dumps(initial_stats) if initial_stats else "null",
    )

    st.html(html, height=72)
