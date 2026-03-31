"""
Structure Viewer Component for IMPULATOR

Renders 2D molecular structures in a floating panel when hovering over Plotly
chart points. Uses OpenChemLib JS (pure JS, no WASM) for client-side SVG
rendering from SMILES.

Security note: All user-visible text is escaped via textContent or the esc()
helper (createElement + textContent). SVG output from OCL.Molecule.toSVG() is
a trusted library output rendered into a dedicated container. innerHTML usage
is limited to trusted, pre-built HTML fragments with escaped user data.

Developed by: Yashwanth Reddy for ITR-UIC
Part of: Chemo-Informatics Toolkit
"""


def get_structure_viewer_component(chart_id="plotly_chart", x_col=None, y_col=None, z_col=None, name_col=None):
    """
    Generate HTML/JS for the hover-based molecular structure viewer.

    Attaches to the nearest Plotly chart and shows a floating SVG structure
    panel on hover (debounced 150ms). Uses OpenChemLib JS v8.18.0.

    Args:
        chart_id: Identifier for the Plotly chart
        x_col: Name of X-axis column (optional)
        y_col: Name of Y-axis column (optional)
        z_col: Name of Z-axis column (optional, for 3D plots)
        name_col: Name of the name/ID column (optional)

    Returns:
        HTML string containing the complete component
    """

    html_component = f"""
    <script>
    (function() {{
        'use strict';

        const CHART_ID = '{chart_id}';
        const parentDoc = window.parent.document;
        const parentWin = window.parent;

        // ── Load OpenChemLib JS ──────────────────────────────────────────
        if (typeof parentWin.OCL === 'undefined') {{
            const script = parentDoc.createElement('script');
            script.src = 'https://unpkg.com/openchemlib@8.18.0/dist/openchemlib-full.js';
            script.onload = () => init();
            script.onerror = () => console.error('[StructViewer] Failed to load OCL');
            parentDoc.head.appendChild(script);
        }} else {{
            init();
        }}

        function init() {{
            injectStyles();
            injectPanel();
            tryAttach(0);
        }}

        // ── CSS ──────────────────────────────────────────────────────────
        function injectStyles() {{
            if (parentDoc.getElementById('sv-style-' + CHART_ID)) return;
            const style = parentDoc.createElement('style');
            style.id = 'sv-style-' + CHART_ID;
            style.textContent = `
                #sv-panel-${{CHART_ID}} {{
                    position: fixed;
                    width: 280px;
                    background: white;
                    border: 1px solid #e0e0e0;
                    border-radius: 10px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
                    z-index: 9999;
                    display: none;
                    flex-direction: column;
                    opacity: 0;
                    transform: scale(0.95);
                    transition: opacity 0.15s ease-out, transform 0.15s ease-out;
                    pointer-events: auto;
                    overflow: hidden;
                }}
                #sv-panel-${{CHART_ID}}.open {{
                    display: flex;
                    opacity: 1;
                    transform: scale(1);
                }}
                #sv-panel-${{CHART_ID}}.pinned {{
                    border-color: #667eea;
                    box-shadow: 0 4px 20px rgba(102,126,234,0.35);
                }}
                #sv-panel-${{CHART_ID}}.dragging {{
                    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
                    opacity: 0.95;
                }}
                .sv-header-${{CHART_ID}} {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 8px 12px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    cursor: grab;
                    user-select: none;
                    flex-shrink: 0;
                }}
                .sv-header-${{CHART_ID}}:active {{ cursor: grabbing; }}
                .sv-header-${{CHART_ID}} h3 {{ margin: 0; font-size: 13px; font-weight: 600; }}
                .sv-close-${{CHART_ID}} {{
                    background: rgba(255,255,255,0.2);
                    border: none; color: white; font-size: 18px;
                    width: 24px; height: 24px; border-radius: 50%;
                    cursor: pointer; display: flex; align-items: center;
                    justify-content: center; padding: 0;
                    transition: background 0.2s;
                }}
                .sv-close-${{CHART_ID}}:hover {{ background: rgba(255,255,255,0.35); }}
                .sv-body-${{CHART_ID}} {{ padding: 10px; }}
                .sv-title-${{CHART_ID}} {{
                    text-align: center; margin-bottom: 6px;
                    padding-bottom: 6px; border-bottom: 1px solid #e9ecef;
                }}
                .sv-name-${{CHART_ID}} {{
                    font-size: 13px; font-weight: 600; color: #212529;
                    margin: 0 0 2px 0; word-break: break-word;
                }}
                .sv-id-${{CHART_ID}} {{
                    font-size: 11px; color: #667eea;
                    font-family: 'Monaco','Menlo','Courier New',monospace; margin: 0;
                }}
                .sv-svg-${{CHART_ID}} {{
                    background: #f8f9fa; border: 1px solid #e9ecef;
                    border-radius: 8px; padding: 8px; margin-bottom: 8px;
                    display: flex; justify-content: center; align-items: center;
                    min-height: 200px;
                }}
                .sv-svg-${{CHART_ID}} svg {{ max-width: 100%; height: auto; }}
                .sv-info-${{CHART_ID}} {{
                    background: #fff; border: 1px solid #dee2e6;
                    border-radius: 6px; padding: 8px; font-size: 11px;
                }}
                .sv-row-${{CHART_ID}} {{ margin-bottom: 6px; }}
                .sv-row-${{CHART_ID}}:last-child {{ margin-bottom: 0; }}
                .sv-lbl-${{CHART_ID}} {{
                    font-weight: 600; color: #495057; font-size: 10px;
                    text-transform: uppercase; letter-spacing: 0.3px; margin-bottom: 2px;
                }}
                .sv-val-${{CHART_ID}} {{
                    font-family: 'Monaco','Menlo','Courier New',monospace;
                    background: #f1f3f5; padding: 4px 8px; border-radius: 4px;
                    font-size: 10px; color: #212529; word-break: break-all;
                    border: 1px solid #e9ecef; max-height: 50px; overflow-y: auto;
                }}
            `;
            parentDoc.head.appendChild(style);
        }}

        // ── Panel HTML ───────────────────────────────────────────────────
        function injectPanel() {{
            if (parentDoc.getElementById('sv-panel-' + CHART_ID)) return;
            const panel = parentDoc.createElement('div');
            panel.id = 'sv-panel-' + CHART_ID;

            // Build panel DOM safely (no innerHTML with user data)
            // Header
            const header = parentDoc.createElement('div');
            header.className = 'sv-header-' + CHART_ID;
            header.id = 'sv-hdr-' + CHART_ID;
            const h3 = parentDoc.createElement('h3');
            h3.textContent = String.fromCodePoint(0x1F9EC) + ' Structure';
            header.appendChild(h3);
            const closeBtn = parentDoc.createElement('button');
            closeBtn.className = 'sv-close-' + CHART_ID;
            closeBtn.id = 'sv-close-' + CHART_ID;
            closeBtn.title = 'Close';
            closeBtn.textContent = 'x';
            header.appendChild(closeBtn);
            panel.appendChild(header);

            // Body
            const body = parentDoc.createElement('div');
            body.className = 'sv-body-' + CHART_ID;

            // Title section
            const titleDiv = parentDoc.createElement('div');
            titleDiv.className = 'sv-title-' + CHART_ID;
            titleDiv.id = 'sv-title-' + CHART_ID;
            titleDiv.style.display = 'none';
            const nameP = parentDoc.createElement('p');
            nameP.className = 'sv-name-' + CHART_ID;
            nameP.id = 'sv-name-' + CHART_ID;
            titleDiv.appendChild(nameP);
            const idP = parentDoc.createElement('p');
            idP.className = 'sv-id-' + CHART_ID;
            idP.id = 'sv-id-' + CHART_ID;
            titleDiv.appendChild(idP);
            body.appendChild(titleDiv);

            // SVG container
            const svgDiv = parentDoc.createElement('div');
            svgDiv.className = 'sv-svg-' + CHART_ID;
            svgDiv.id = 'sv-svg-' + CHART_ID;
            const placeholder = parentDoc.createElement('span');
            placeholder.style.cssText = 'color:#adb5bd;font-size:12px;';
            placeholder.textContent = 'Hover a data point';
            svgDiv.appendChild(placeholder);
            body.appendChild(svgDiv);

            // Info section
            const infoDiv = parentDoc.createElement('div');
            infoDiv.className = 'sv-info-' + CHART_ID;
            infoDiv.id = 'sv-info-' + CHART_ID;
            body.appendChild(infoDiv);

            panel.appendChild(body);
            parentDoc.body.appendChild(panel);

            // Close button handler
            closeBtn.onclick = (e) => {{
                e.preventDefault(); e.stopPropagation();
                panel.classList.remove('open');
                panel.classList.remove('pinned');
            }};

            // Drag support
            let dragging = false, dx = 0, dy = 0, px = 0, py = 0;

            header.addEventListener('mousedown', (e) => {{
                if (e.target.tagName === 'BUTTON') return;
                dragging = true; panel.classList.add('dragging');
                dx = e.clientX; dy = e.clientY;
                const r = panel.getBoundingClientRect();
                px = r.left; py = r.top;
                e.preventDefault();
            }});
            parentDoc.addEventListener('mousemove', (e) => {{
                if (!dragging) return;
                const nx = px + (e.clientX - dx);
                const ny = py + (e.clientY - dy);
                panel.style.left = Math.max(0, Math.min(nx, parentWin.innerWidth - panel.offsetWidth)) + 'px';
                panel.style.top = Math.max(0, Math.min(ny, parentWin.innerHeight - panel.offsetHeight)) + 'px';
            }});
            parentDoc.addEventListener('mouseup', () => {{
                if (dragging) {{ dragging = false; panel.classList.remove('dragging'); }}
            }});
        }}

        // ── Find nearest Plotly chart ────────────────────────────────────
        function findChart() {{
            // Strategy 1: Find our iframe, then closest chart above it
            let ourIframe = null;
            for (const iframe of parentDoc.querySelectorAll('iframe')) {{
                try {{ if (iframe.contentWindow === window) {{ ourIframe = iframe; break; }} }}
                catch (e) {{}}
            }}

            if (ourIframe) {{
                const charts = Array.from(parentDoc.querySelectorAll('.js-plotly-plot'));
                const iRect = ourIframe.getBoundingClientRect();
                let best = null, bestDist = Infinity;

                for (const c of charts) {{
                    const cRect = c.getBoundingClientRect();
                    if (cRect.bottom <= iRect.top + 50) {{
                        const d = iRect.top - cRect.bottom;
                        if (d < bestDist) {{ bestDist = d; best = c; }}
                    }}
                }}
                if (best) return best;

                // Fallback: sibling walk
                let el = ourIframe.closest('.stHtml, [data-testid="stHtml"], .element-container, div');
                if (el) {{
                    let sib = el.previousElementSibling;
                    while (sib) {{
                        const p = sib.querySelector('.js-plotly-plot');
                        if (p) return p;
                        sib = sib.previousElementSibling;
                    }}
                }}
            }}

            // Strategy 2: Key-based search
            const containers = parentDoc.querySelectorAll('[data-testid="stPlotlyChart"]');
            for (const cont of containers) {{
                let el = cont;
                while (el && el !== parentDoc.body) {{
                    const k = el.getAttribute('key') || el.getAttribute('data-key') || el.getAttribute('id') || '';
                    if (k.includes(CHART_ID)) {{
                        const p = cont.querySelector('.js-plotly-plot');
                        if (p) return p;
                    }}
                    el = el.parentElement;
                }}
            }}

            // Strategy 3: Any unattached chart with customdata
            for (const div of parentDoc.querySelectorAll('.js-plotly-plot')) {{
                if (!div['_sv_' + CHART_ID] && div._fullData && div._fullData[0] && div._fullData[0].customdata) {{
                    return div;
                }}
            }}

            return null;
        }}

        // ── Attach hover listeners ───────────────────────────────────────
        function attachListeners() {{
            const plotlyDiv = findChart();
            if (!plotlyDiv || !parentWin.Plotly) return false;
            if (plotlyDiv['_sv_' + CHART_ID]) return true;
            plotlyDiv['_sv_' + CHART_ID] = true;

            const panel = parentDoc.getElementById('sv-panel-' + CHART_ID);
            const svgContainer = parentDoc.getElementById('sv-svg-' + CHART_ID);
            const infoDiv = parentDoc.getElementById('sv-info-' + CHART_ID);
            const titleDiv = parentDoc.getElementById('sv-title-' + CHART_ID);
            const nameEl = parentDoc.getElementById('sv-name-' + CHART_ID);
            const idEl = parentDoc.getElementById('sv-id-' + CHART_ID);

            let hoverTimer = null;
            let lastSmiles = '';
            let mouseX = 0, mouseY = 0;

            // Track mouse for panel positioning
            plotlyDiv.addEventListener('mousemove', (e) => {{
                mouseX = e.clientX; mouseY = e.clientY;
            }}, true);

            // Column names for display
            const cols = {{
                x: {repr(x_col) if x_col else 'null'},
                y: {repr(y_col) if y_col else 'null'},
                z: {repr(z_col) if z_col else 'null'},
                name: {repr(name_col) if name_col else 'null'}
            }};

            // ── Hover: show preview (auto-hides on unhover) ─────────────
            plotlyDiv.on('plotly_hover', function(data) {{
                if (!data.points || data.points.length === 0) return;
                if (panel.classList.contains('pinned')) return; // Don't override pinned panel
                const pt = data.points[0];

                clearTimeout(hoverTimer);
                hoverTimer = setTimeout(() => {{
                    if (!pt.customdata) return;
                    const smiles = pt.customdata[0];
                    if (!smiles || smiles === 'null') return;

                    if (smiles !== lastSmiles) {{
                        lastSmiles = smiles;
                        let molName = null, molId = null;
                        if (pt.customdata.length >= 3) {{ molName = pt.customdata[1]; molId = pt.customdata[2]; }}
                        else if (pt.customdata.length === 2) {{ molName = pt.customdata[1]; }}
                        renderOCL(smiles, molName, molId, pt);
                    }}
                    positionPanel(mouseX, mouseY);
                    panel.classList.add('open');
                }}, 150);
            }});

            // ── Unhover: hide unless pinned ──────────────────────────────
            plotlyDiv.on('plotly_unhover', function() {{
                clearTimeout(hoverTimer);
                hoverTimer = null;
                if (panel.classList.contains('pinned')) return; // Stay open if pinned
                setTimeout(() => {{
                    if (!hoverTimer && !panel.classList.contains('pinned')) {{
                        panel.classList.remove('open');
                    }}
                }}, 200);
            }});

            // ── Click: pin the panel (stays until X or click outside) ────
            plotlyDiv.on('plotly_click', function(data) {{
                if (!data.points || data.points.length === 0) return;
                const pt = data.points[0];
                if (!pt.customdata) return;
                const smiles = pt.customdata[0];
                if (!smiles || smiles === 'null') return;

                // If clicking same molecule that's already pinned, unpin
                if (panel.classList.contains('pinned') && smiles === lastSmiles) {{
                    panel.classList.remove('pinned');
                    panel.classList.remove('open');
                    lastSmiles = '';
                    return;
                }}

                lastSmiles = smiles;
                let molName = null, molId = null;
                if (pt.customdata.length >= 3) {{ molName = pt.customdata[1]; molId = pt.customdata[2]; }}
                else if (pt.customdata.length === 2) {{ molName = pt.customdata[1]; }}

                renderOCL(smiles, molName, molId, pt);
                positionPanel(mouseX, mouseY);
                panel.classList.add('open');
                panel.classList.add('pinned');
            }});

            // ── Click outside: unpin and close ───────────────────────────
            parentDoc.addEventListener('mousedown', function(e) {{
                if (!panel.classList.contains('pinned')) return;
                // Check if click is inside the panel
                if (panel.contains(e.target)) return;
                // Check if click is inside the plotly chart (let plotly_click handle it)
                if (plotlyDiv.contains(e.target)) return;
                panel.classList.remove('pinned');
                panel.classList.remove('open');
                lastSmiles = '';
            }});

            // ── Render molecule with OpenChemLib ─────────────────────────
            function renderOCL(smiles, molName, molId, pointData) {{
                try {{
                    const mol = parentWin.OCL.Molecule.fromSmiles(smiles);
                    // OCL.Molecule.toSVG() returns trusted SVG from the library
                    const svgStr = mol.toSVG(240, 200, null, {{
                        suppressChiralText: true,
                        suppressESR: true,
                        noStereoProblem: true,
                    }});
                    // Clear and set SVG (trusted library output, not user input)
                    while (svgContainer.firstChild) svgContainer.removeChild(svgContainer.firstChild);
                    const wrapper = parentDoc.createElement('div');
                    wrapper.insertAdjacentHTML('afterbegin', svgStr);
                    if (wrapper.firstChild) svgContainer.appendChild(wrapper.firstChild);
                }} catch (err) {{
                    while (svgContainer.firstChild) svgContainer.removeChild(svgContainer.firstChild);
                    const errSpan = parentDoc.createElement('span');
                    errSpan.style.cssText = 'color:#c33;font-size:11px;';
                    errSpan.textContent = 'Invalid SMILES';
                    svgContainer.appendChild(errSpan);
                }}

                // Update title (using textContent for safety)
                if (molName) {{
                    const nameStr = String(molName);
                    const isChembl = nameStr.startsWith('CHEMBL');
                    if (isChembl) {{
                        nameEl.textContent = '';
                        nameEl.style.display = 'none';
                        idEl.textContent = nameStr;
                        idEl.style.display = 'block';
                    }} else {{
                        nameEl.textContent = nameStr;
                        nameEl.style.display = 'block';
                        if (molId && String(molId).startsWith('CHEMBL')) {{
                            idEl.textContent = String(molId);
                            idEl.style.display = 'block';
                        }} else {{
                            idEl.style.display = 'none';
                        }}
                    }}
                    titleDiv.style.display = 'block';
                }} else {{
                    titleDiv.style.display = 'none';
                }}

                // Update info section (build DOM safely)
                while (infoDiv.firstChild) infoDiv.removeChild(infoDiv.firstChild);
                appendInfoRow(infoDiv, 'SMILES', smiles);
                if (pointData.x !== undefined) {{
                    const lbl = cols.x ? 'X (' + cols.x + ')' : 'X';
                    appendInfoRow(infoDiv, lbl, fmt(pointData.x));
                }}
                if (pointData.y !== undefined) {{
                    const lbl = cols.y ? 'Y (' + cols.y + ')' : 'Y';
                    appendInfoRow(infoDiv, lbl, fmt(pointData.y));
                }}
                if (pointData.z !== undefined) {{
                    const lbl = cols.z ? 'Z (' + cols.z + ')' : 'Z';
                    appendInfoRow(infoDiv, lbl, fmt(pointData.z));
                }}
            }}

            // Build an info row using safe DOM methods
            function appendInfoRow(container, label, value) {{
                const row = parentDoc.createElement('div');
                row.className = 'sv-row-' + CHART_ID;
                const lblDiv = parentDoc.createElement('div');
                lblDiv.className = 'sv-lbl-' + CHART_ID;
                lblDiv.textContent = label;
                const valDiv = parentDoc.createElement('div');
                valDiv.className = 'sv-val-' + CHART_ID;
                valDiv.textContent = value;
                row.appendChild(lblDiv);
                row.appendChild(valDiv);
                container.appendChild(row);
            }}

            // ── Position panel near cursor ───────────────────────────────
            function positionPanel(cx, cy) {{
                const pw = 280, ph = 420, pad = 15, edge = 10;
                const vw = parentWin.innerWidth, vh = parentWin.innerHeight;
                let left = cx + pad, top = cy + pad;
                if (left + pw > vw - edge) left = cx - pw - pad;
                if (left < edge) left = edge;
                if (top + ph > vh - edge) top = cy - ph - pad;
                if (top < edge) top = edge;
                panel.style.left = left + 'px';
                panel.style.top = top + 'px';
                panel.style.right = 'auto';
            }}

            return true;
        }}

        // ── Helpers ──────────────────────────────────────────────────────
        function fmt(num) {{
            return typeof num === 'number' ? num.toFixed(3) : String(num);
        }}

        // ── Retry loop ───────────────────────────────────────────────────
        function tryAttach(attempt) {{
            if (attachListeners()) return;
            if (attempt < 10) {{
                setTimeout(() => tryAttach(attempt + 1), 500);
            }}
        }}

        // ── Re-attach on Streamlit updates ───────────────────────────────
        let debounce;
        const observer = new MutationObserver(() => {{
            clearTimeout(debounce);
            debounce = setTimeout(() => {{
                const chart = findChart();
                if (chart && !chart['_sv_' + CHART_ID]) attachListeners();
            }}, 100);
        }});

        const container = parentDoc.querySelector('[data-testid="stAppViewContainer"]');
        if (container) {{
            observer.observe(container, {{ childList: true, subtree: true }});
        }}

        // ── Cleanup when iframe is removed (toggle off / Streamlit rerender) ─
        // When the toggle is turned off, Streamlit removes this iframe but the
        // panel/style we injected into parentDoc stay behind. Watch for our
        // iframe disappearing and clean up.
        function cleanup() {{
            const p = parentDoc.getElementById('sv-panel-' + CHART_ID);
            if (p) p.remove();
            const s = parentDoc.getElementById('sv-style-' + CHART_ID);
            if (s) s.remove();
            observer.disconnect();
            // Remove hover flag from chart so it can be re-attached later
            const chart = findChart();
            if (chart) delete chart['_sv_' + CHART_ID];
        }}

        // Detect iframe removal via periodic check (MutationObserver can't
        // observe its own removal). Runs every 2s, very lightweight.
        const aliveCheck = setInterval(() => {{
            let found = false;
            for (const iframe of parentDoc.querySelectorAll('iframe')) {{
                try {{ if (iframe.contentWindow === window) {{ found = true; break; }} }}
                catch (e) {{}}
            }}
            if (!found) {{
                clearInterval(aliveCheck);
                cleanup();
            }}
        }}, 2000);

        // Also clean up on page unload
        window.addEventListener('unload', () => {{
            clearInterval(aliveCheck);
            cleanup();
        }});
    }})();
    </script>
    """

    return html_component


def get_structure_viewer_hint():
    """Hint message for charts with hover-to-view structure support."""
    return """
    <div style="
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        border: 1px solid #667eea;
        border-radius: 6px;
        padding: 10px 15px;
        margin: 10px 0;
        font-size: 13px;
        color: #5a67d8;
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        <span style="font-size: 16px;">&#x1F9EC;</span>
        <span><strong>Tip:</strong> Hover to preview, click to pin the 2D molecular structure</span>
    </div>
    """
