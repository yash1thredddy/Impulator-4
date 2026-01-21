"""
Structure Viewer Component for Molecular Properties Calculator

This module provides a client-side 2D molecular structure viewer using SmilesDrawer.
When users click on plot points, the molecular structure is displayed in a collapsible
side panel without triggering Streamlit reruns.

Developed by: Yashwanth Reddy for ITR-UIC
Part of: Chemo-Informatics Toolkit
"""


def get_structure_viewer_component(chart_id="plotly_chart", x_col=None, y_col=None, z_col=None, name_col=None):
    """
    Generate HTML/CSS/JS for the interactive structure viewer component.
    
    This component:
    - Displays a collapsible side panel for molecular structures
    - Uses SmilesDrawer to render 2D structures on HTML5 canvas
    - Listens to Plotly click events without triggering Streamlit reruns
    - Maintains data integrity across filtering and plot updates
    
    Args:
        chart_id: Identifier for the Plotly chart (for targeting specific charts)
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
        
        console.log('[Structure Viewer {chart_id}] Initializing component...');
        
        const parentDoc = window.parent.document;
        const parentWin = window.parent;
        
        // Load SmilesDrawer from CDN if not already loaded
        if (typeof parentWin.SmilesDrawer === 'undefined') {{
            console.log('[Structure Viewer {chart_id}] Loading SmilesDrawer from jsDelivr CDN...');
            const script = parentDoc.createElement('script');
            // Use jsDelivr CDN which is less likely to be blocked by tracking prevention
            
            // TODO: Upgrade to SmilesDrawer 2.1.7+ when compatibility issues are resolved
            // PINNED TO v2.0.1 - DO NOT UPGRADE WITHOUT TESTING
            // Reason: Version 2.1.7 has breaking changes that prevent molecule rendering
            // Issue: Molecules render as blank canvases with v2.1.7 (tested 2025-12-09)
            // Blockers:
            //   - API changes in parse() or draw() methods
            //   - Constructor signature changes
            //   - Canvas rendering pipeline changes
            // Next steps:
            //   1. Check SmilesDrawer GitHub changelog for breaking changes
            //   2. Test intermediate versions (2.1.0-2.1.6) to identify breaking version
            //   3. Update our code to match new API if needed
            //   4. Comprehensive testing before upgrading
            // Revisit: Q1 2026 or when official migration guide is available
            script.src = 'https://cdn.jsdelivr.net/npm/smiles-drawer@2.0.1/dist/smiles-drawer.min.js';
            script.onload = function() {{
                console.log('[Structure Viewer {chart_id}] ✅ SmilesDrawer v2.0.1 loaded successfully from jsDelivr');
                initViewer();
            }};
            script.onerror = function(err) {{
                console.error('[Structure Viewer {chart_id}] ❌ Failed to load SmilesDrawer from CDN:', err);
                console.error('[Structure Viewer {chart_id}] Try disabling tracking prevention in your browser');
            }};
            parentDoc.head.appendChild(script);
        }} else {{
            console.log('[Structure Viewer {chart_id}] SmilesDrawer already loaded');
            initViewer();
        }}
        
        function initViewer() {{
            // Inject CSS styles into parent document
            if (!parentDoc.getElementById('structure-viewer-style-{chart_id}')) {{
                const style = parentDoc.createElement('style');
                style.id = 'structure-viewer-style-{chart_id}';
                style.textContent = `
                    #structure-panel-{chart_id} {{
                        position: fixed;
                        top: 80px;
                        right: 10px;
                        width: 340px;
                        max-height: 520px;
                        background: white;
                        border: 1px solid #e0e0e0;
                        border-radius: 10px;
                        box-shadow: 0 4px 16px rgba(0,0,0,0.18);
                        z-index: 9999;
                        transform: translateX(calc(100% + 20px));
                        transition: transform 0.3s ease-in-out, box-shadow 0.2s;
                        overflow-y: auto;
                        display: flex;
                        flex-direction: column;
                        cursor: default;
                    }}

                    #structure-panel-{chart_id}.open {{
                        transform: translateX(0);
                    }}

                    #structure-panel-{chart_id}.dragging {{
                        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
                        opacity: 0.95;
                    }}

                    .panel-header-{chart_id} {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 12px 16px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        border-bottom: 1px solid #5a67d8;
                        border-radius: 10px 10px 0 0;
                        flex-shrink: 0;
                        cursor: grab;
                        user-select: none;
                    }}

                    .panel-header-{chart_id}:active {{
                        cursor: grabbing;
                    }}

                    .panel-header-{chart_id} h3 {{
                        margin: 0;
                        font-size: 15px;
                        font-weight: 600;
                    }}

                    .close-btn-{chart_id} {{
                        background: rgba(255,255,255,0.2);
                        border: none;
                        color: white;
                        font-size: 22px;
                        width: 28px;
                        height: 28px;
                        border-radius: 50%;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        line-height: 1;
                        transition: all 0.2s;
                        padding: 0;
                    }}

                    .close-btn-{chart_id}:hover {{
                        background: rgba(255,255,255,0.3);
                        transform: scale(1.1);
                    }}

                    .panel-content-{chart_id} {{
                        padding: 14px;
                        flex: 1;
                        overflow-y: auto;
                    }}

                    .molecule-title-{chart_id} {{
                        text-align: center;
                        margin-bottom: 10px;
                        padding-bottom: 10px;
                        border-bottom: 1px solid #e9ecef;
                    }}

                    .molecule-name-{chart_id} {{
                        font-size: 14px;
                        font-weight: 600;
                        color: #212529;
                        margin: 0 0 4px 0;
                        word-break: break-word;
                    }}

                    .molecule-id-{chart_id} {{
                        font-size: 12px;
                        color: #667eea;
                        font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
                        margin: 0;
                    }}

                    .structure-canvas-container-{chart_id} {{
                        background: #f8f9fa;
                        border: 1px solid #e9ecef;
                        border-radius: 8px;
                        padding: 10px;
                        margin-bottom: 12px;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 260px;
                    }}

                    #structure-canvas-{chart_id} {{
                        max-width: 100%;
                        height: auto;
                    }}

                    .structure-info-{chart_id} {{
                        background: #ffffff;
                        border: 1px solid #dee2e6;
                        border-radius: 6px;
                        padding: 12px;
                    }}

                    .info-row-{chart_id} {{
                        margin-bottom: 10px;
                        display: flex;
                        flex-direction: column;
                    }}

                    .info-row-{chart_id}:last-child {{
                        margin-bottom: 0;
                    }}

                    .info-label-{chart_id} {{
                        font-weight: 600;
                        color: #495057;
                        font-size: 11px;
                        margin-bottom: 3px;
                        text-transform: uppercase;
                        letter-spacing: 0.3px;
                    }}

                    .info-value-{chart_id} {{
                        font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
                        background: #f1f3f5;
                        padding: 6px 10px;
                        border-radius: 4px;
                        font-size: 11px;
                        color: #212529;
                        word-break: break-all;
                        border: 1px solid #e9ecef;
                        max-height: 60px;
                        overflow-y: auto;
                    }}
                    
                    .error-message-{chart_id} {{
                        background: #fee;
                        border: 1px solid #fcc;
                        color: #c33;
                        padding: 12px;
                        border-radius: 6px;
                        margin-top: 12px;
                        font-size: 13px;
                    }}
                    
                    .loading-message-{chart_id} {{
                        text-align: center;
                        padding: 40px 20px;
                        color: #6c757d;
                        font-size: 14px;
                    }}
                `;
                parentDoc.head.appendChild(style);
                console.log('[Structure Viewer {chart_id}] ✅ Style injected');
            }}
            
            // Inject panel HTML into parent document
            if (!parentDoc.getElementById('structure-panel-{chart_id}')) {{
                const panelHTML = `
                <div id="structure-panel-{chart_id}">
                    <div class="panel-header-{chart_id}" id="panel-header-{chart_id}">
                        <h3>🧬 Molecular Structure</h3>
                        <button class="close-btn-{chart_id}" id="close-panel-btn-{chart_id}" title="Close panel">×</button>
                    </div>
                    <div class="panel-content-{chart_id}">
                        <div class="molecule-title-{chart_id}" id="molecule-title-{chart_id}" style="display: none;">
                            <p class="molecule-name-{chart_id}" id="molecule-name-{chart_id}"></p>
                            <p class="molecule-id-{chart_id}" id="molecule-id-{chart_id}"></p>
                        </div>
                        <div class="structure-canvas-container-{chart_id}">
                            <canvas id="structure-canvas-{chart_id}" width="220" height="220"></canvas>
                        </div>
                        <div class="structure-info-{chart_id}" id="structure-info-{chart_id}">
                            <div class="loading-message-{chart_id}">
                                Click on a data point to view its molecular structure
                            </div>
                        </div>
                    </div>
                </div>
                `;
                
                const panelContainer = parentDoc.createElement('div');
                panelContainer.innerHTML = panelHTML;
                parentDoc.body.appendChild(panelContainer.firstElementChild);
                console.log('[Structure Viewer {chart_id}] ✅ Panel injected into parent document');
            }} else {{
                console.log('[Structure Viewer {chart_id}] Panel already exists');
            }}
            
            // Get references to panel elements
            const panel = parentDoc.getElementById('structure-panel-{chart_id}');
            const closeBtn = parentDoc.getElementById('close-panel-btn-{chart_id}');
            const canvas = parentDoc.getElementById('structure-canvas-{chart_id}');
            const infoDiv = parentDoc.getElementById('structure-info-{chart_id}');
            
            if (!panel || !canvas) {{
                console.error('[Structure Viewer {chart_id}] Failed to create panel elements');
                return;
            }}
            
            console.log('[Structure Viewer {chart_id}] ✅ Panel elements ready');
            
            // Initialize SmilesDrawer
            let drawer = null;
            if (typeof parentWin.SmilesDrawer !== 'undefined') {{
                console.log('[Structure Viewer {chart_id}] SmilesDrawer object:', Object.keys(parentWin.SmilesDrawer));
                
                // v2.0.1 uses SmilesDrawer.Drawer constructor
                // Note: v2.1.7+ may use different constructor name
                if (parentWin.SmilesDrawer.Drawer) {{
                    drawer = new parentWin.SmilesDrawer.Drawer({{
                        width: 220,
                        height: 220,
                        bondThickness: 1.2,
                        fontFamily: 'Arial, sans-serif',
                        fontSize: 12
                    }});
                    console.log('[Structure Viewer {chart_id}] ✅ SmilesDrawer v2.0.1 initialized');
                }} else {{
                    console.error('[Structure Viewer {chart_id}] ❌ Drawer constructor not found');
                    console.error('[Structure Viewer {chart_id}] Available constructors:', Object.keys(parentWin.SmilesDrawer));
                    return;
                }}
            }} else {{
                console.error('[Structure Viewer {chart_id}] ❌ SmilesDrawer library not loaded');
                return;
            }}
            
            // Close button handler (CRITICAL: prevent reruns)
            closeBtn.onclick = function(e) {{
                e.preventDefault();
                e.stopPropagation();
                panel.classList.remove('open');
                console.log('[Structure Viewer {chart_id}] Panel closed');
                return false;
            }};

            // Get references to molecule title elements
            const moleculeTitleDiv = parentDoc.getElementById('molecule-title-{chart_id}');
            const moleculeNameEl = parentDoc.getElementById('molecule-name-{chart_id}');
            const moleculeIdEl = parentDoc.getElementById('molecule-id-{chart_id}');
            const panelHeader = parentDoc.getElementById('panel-header-{chart_id}');

            // Drag functionality
            let isDragging = false;
            let dragStartX = 0;
            let dragStartY = 0;
            let panelStartX = 0;
            let panelStartY = 0;

            panelHeader.addEventListener('mousedown', function(e) {{
                // Don't start drag if clicking close button
                if (e.target === closeBtn) return;

                isDragging = true;
                panel.classList.add('dragging');

                dragStartX = e.clientX;
                dragStartY = e.clientY;

                const rect = panel.getBoundingClientRect();
                panelStartX = rect.left;
                panelStartY = rect.top;

                // Remove transform and set position directly
                panel.style.transform = 'none';
                panel.style.left = rect.left + 'px';
                panel.style.top = rect.top + 'px';
                panel.style.right = 'auto';

                e.preventDefault();
            }});

            parentDoc.addEventListener('mousemove', function(e) {{
                if (!isDragging) return;

                const deltaX = e.clientX - dragStartX;
                const deltaY = e.clientY - dragStartY;

                const newX = panelStartX + deltaX;
                const newY = panelStartY + deltaY;

                // Constrain to viewport
                const maxX = parentWin.innerWidth - panel.offsetWidth;
                const maxY = parentWin.innerHeight - panel.offsetHeight;

                panel.style.left = Math.max(0, Math.min(newX, maxX)) + 'px';
                panel.style.top = Math.max(0, Math.min(newY, maxY)) + 'px';
            }});

            parentDoc.addEventListener('mouseup', function() {{
                if (isDragging) {{
                    isDragging = false;
                    panel.classList.remove('dragging');
                }}
            }});

            // Find Plotly chart
            function findPlotlyChart() {{
                const selectors = [
                    '.js-plotly-plot',
                    '.plotly-graph-div.js-plotly-plot',
                    '[data-testid="stPlotlyChart"] .js-plotly-plot'
                ];
                
                for (const selector of selectors) {{
                    const elements = parentDoc.querySelectorAll(selector);
                    if (elements.length > 0) {{
                        return elements[elements.length - 1];
                    }}
                }}
                
                // Fallback: find by _fullLayout property
                const allDivs = parentDoc.querySelectorAll('div');
                for (let div of allDivs) {{
                    if (div._fullLayout) {{
                        return div;
                    }}
                }}
                
                return null;
            }}
            
            // Attach click listener
            function attachClickListener() {{
                const plotlyDiv = findPlotlyChart();
                
                if (!plotlyDiv) {{
                    console.warn('[Structure Viewer {chart_id}] Plotly chart not found, retrying...');
                    setTimeout(attachClickListener, 500);
                    return;
                }}
                
                if (typeof parentWin.Plotly === 'undefined') {{
                    console.warn('[Structure Viewer {chart_id}] Plotly library not loaded, retrying...');
                    setTimeout(attachClickListener, 500);
                    return;
                }}
                
                if (plotlyDiv._structureViewerAttached_{chart_id}) {{
                    console.log('[Structure Viewer {chart_id}] Listener already attached');
                    return;
                }}
                plotlyDiv._structureViewerAttached_{chart_id} = true;
                
                console.log('[Structure Viewer {chart_id}] ✅ Attaching click listener');
                
                // Listen for Plotly click events
                plotlyDiv.on('plotly_click', function(data) {{
                    console.log('[Structure Viewer {chart_id}] 🖱️ Click detected!', data);
                    
                    if (!data.points || data.points.length === 0) {{
                        console.warn('[Structure Viewer {chart_id}] No points in click data');
                        return;
                    }}
                    
                    const point = data.points[0];
                    console.log('[Structure Viewer {chart_id}] Point data:', point);
                    
                    // Extract data from customdata
                    // Format can be: [SMILES, name, index] or [SMILES, index]
                    let smiles = null;
                    let moleculeName = null;
                    let pointIndex = null;
                    
                    if (point.customdata) {{
                        console.log('[Structure Viewer {chart_id}] customdata:', point.customdata);
                        smiles = point.customdata[0];
                        
                        // Check format: if length is 3, we have [SMILES, name, index]
                        if (point.customdata.length === 3) {{
                            moleculeName = point.customdata[1];
                            pointIndex = point.customdata[2];
                        }} else if (point.customdata.length === 2) {{
                            // Format: [SMILES, index]
                            pointIndex = point.customdata[1];
                        }}
                    }} else {{
                        console.warn('[Structure Viewer {chart_id}] No customdata found');
                    }}
                    
                    if (!smiles || smiles === null || smiles === 'null') {{
                        console.warn('[Structure Viewer {chart_id}] Invalid SMILES:', smiles);
                        infoDiv.innerHTML = `
                            <div class="error-message-{chart_id}">
                                No SMILES data available for this point<br>
                                <small>Debug: ${{JSON.stringify(point.customdata || 'undefined')}}</small>
                            </div>
                        `;
                        panel.classList.add('open');
                        return;
                    }}
                    
                    console.log('[Structure Viewer {chart_id}] Rendering SMILES:', smiles, 'Name:', moleculeName);
                    
                    // Pass column names for display
                    const columnNames = {{
                        x: {repr(x_col) if x_col else 'null'},
                        y: {repr(y_col) if y_col else 'null'},
                        z: {repr(z_col) if z_col else 'null'},
                        name: {repr(name_col) if name_col else 'null'}
                    }};
                    
                    renderStructure(smiles, moleculeName, pointIndex, point, columnNames);
                }});
                
                console.log('[Structure Viewer {chart_id}] ✅ Click listener attached successfully');
            }}
            
            // Render molecular structure
            function renderStructure(smiles, moleculeName, pointIndex, pointData, columnNames) {{
                console.log('[Structure Viewer {chart_id}] renderStructure called for:', smiles);
                
                infoDiv.innerHTML = '<div class="loading-message-{chart_id}">Rendering structure...</div>';
                panel.classList.add('open');
                
                // Basic SMILES validation
                if (!smiles || typeof smiles !== 'string' || smiles.trim() === '') {{
                    console.error('[Structure Viewer {chart_id}] Invalid SMILES: empty or not a string');
                    showError('Empty or invalid SMILES string', smiles, moleculeName);
                    return;
                }}
                
                // Check for obviously malformed SMILES (basic heuristics)
                const trimmedSmiles = smiles.trim();
                if (trimmedSmiles.length > 1000) {{
                    console.warn('[Structure Viewer {chart_id}] SMILES unusually long (>1000 chars)');
                }}
                
                try {{
                    parentWin.SmilesDrawer.parse(smiles, function(tree) {{
                        if (!tree) {{
                            console.error('[Structure Viewer {chart_id}] Parse returned null/undefined tree');
                            showError('Failed to parse SMILES - structure may be invalid', smiles, moleculeName);
                            return;
                        }}
                        
                        console.log('[Structure Viewer {chart_id}] SMILES parsed, drawing...');
                        
                        // Clear canvas
                        const ctx = canvas.getContext('2d');
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        
                        // Draw structure
                        drawer.draw(tree, canvas, 'light', false);
                        
                        console.log('[Structure Viewer {chart_id}] ✅ Structure drawn');

                        // Update molecule title section (prominent display of name and ChEMBL ID)
                        if (moleculeName !== null && moleculeName !== undefined && moleculeName !== '') {{
                            const nameStr = String(moleculeName);
                            // Check if this is a ChEMBL ID
                            const isChemblId = nameStr.startsWith('CHEMBL');

                            if (isChemblId) {{
                                // Show only ChEMBL ID
                                moleculeNameEl.textContent = '';
                                moleculeNameEl.style.display = 'none';
                                moleculeIdEl.textContent = nameStr;
                                moleculeIdEl.style.display = 'block';
                            }} else {{
                                // Show both name and potentially infer ChEMBL ID from customdata
                                moleculeNameEl.textContent = nameStr;
                                moleculeNameEl.style.display = 'block';
                                // Check if there's a ChEMBL ID in the data
                                if (pointData.customdata && pointData.customdata.length > 1) {{
                                    const possibleId = String(pointData.customdata[1] || '');
                                    if (possibleId.startsWith('CHEMBL')) {{
                                        moleculeIdEl.textContent = possibleId;
                                        moleculeIdEl.style.display = 'block';
                                    }} else {{
                                        moleculeIdEl.style.display = 'none';
                                    }}
                                }} else {{
                                    moleculeIdEl.style.display = 'none';
                                }}
                            }}
                            moleculeTitleDiv.style.display = 'block';
                        }} else {{
                            moleculeTitleDiv.style.display = 'none';
                        }}

                        // Update info panel - coordinates and SMILES
                        let infoHTML = '';

                        infoHTML += `
                            <div class="info-row-{chart_id}">
                                <div class="info-label-{chart_id}">SMILES</div>
                                <div class="info-value-{chart_id}">${{escapeHtml(smiles)}}</div>
                            </div>
                        `;
                        
                        if (pointData.x !== undefined) {{
                            const xLabel = columnNames.x ? `X (${{escapeHtml(columnNames.x)}})` : 'X Value';
                            infoHTML += `
                                <div class="info-row-{chart_id}">
                                    <div class="info-label-{chart_id}">${{xLabel}}</div>
                                    <div class="info-value-{chart_id}">${{formatNumber(pointData.x)}}</div>
                                </div>
                            `;
                        }}
                        
                        if (pointData.y !== undefined) {{
                            const yLabel = columnNames.y ? `Y (${{escapeHtml(columnNames.y)}})` : 'Y Value';
                            infoHTML += `
                                <div class="info-row-{chart_id}">
                                    <div class="info-label-{chart_id}">${{yLabel}}</div>
                                    <div class="info-value-{chart_id}">${{formatNumber(pointData.y)}}</div>
                                </div>
                            `;
                        }}
                        
                        if (pointData.z !== undefined) {{
                            const zLabel = columnNames.z ? `Z (${{escapeHtml(columnNames.z)}})` : 'Z Value';
                            infoHTML += `
                                <div class="info-row-{chart_id}">
                                    <div class="info-label-{chart_id}">${{zLabel}}</div>
                                    <div class="info-value-{chart_id}">${{formatNumber(pointData.z)}}</div>
                                </div>
                            `;
                        }}
                        
                        infoDiv.innerHTML = infoHTML;
                    }}, function(err) {{
                        console.error('[Structure Viewer {chart_id}] Error parsing SMILES:', err);
                        showError('Invalid SMILES string - parse failed', smiles, moleculeName, err);
                    }});
                }} catch (error) {{
                    console.error('[Structure Viewer {chart_id}] Error rendering:', error);
                    showError('Failed to render structure', smiles, moleculeName, error);
                }}
            }}
            
            // Show error message with helpful context
            function showError(message, smiles, moleculeName, error = null) {{
                let errorHTML = `
                    <div class="error-message-{chart_id}">
                        <strong>⚠️ ${{escapeHtml(message)}}</strong><br><br>
                `;
                
                // Show molecule identification
                if (moleculeName) {{
                    errorHTML += `
                        <div style="margin: 10px 0; padding: 8px; background: rgba(0,0,0,0.05); border-radius: 4px;">
                            <strong>Molecule:</strong> ${{escapeHtml(String(moleculeName))}}
                        </div>
                    `;
                }}
                
                // Show SMILES with truncation if too long
                const maxSmilesDisplay = 100;
                const displaySmiles = smiles && smiles.length > maxSmilesDisplay 
                    ? smiles.substring(0, maxSmilesDisplay) + '...' 
                    : smiles;
                    
                errorHTML += `
                    <div style="margin: 10px 0; padding: 8px; background: rgba(0,0,0,0.05); border-radius: 4px;">
                        <strong>SMILES:</strong><br>
                        <code style="font-size: 11px; word-break: break-all;">${{escapeHtml(displaySmiles || 'N/A')}}</code>
                    </div>
                `;
                
                // Show technical error if available
                if (error && error.message) {{
                    errorHTML += `
                        <div style="margin: 10px 0; font-size: 11px; color: #666;">
                            <strong>Technical details:</strong> ${{escapeHtml(error.message)}}
                        </div>
                    `;
                }}
                
                errorHTML += `
                    <div style="margin-top: 15px; padding: 10px; background: #fff3cd; border-left: 3px solid #ffc107; font-size: 12px;">
                        <strong>💡 Suggestions:</strong><br>
                        • Check if the SMILES string is complete and valid<br>
                        • Verify the source data doesn't have truncated/corrupted values<br>
                        • Try validating the SMILES with an external tool<br>
                        • Contact support if this persists for valid structures
                    </div>
                    </div>
                `;
                
                infoDiv.innerHTML = errorHTML;
            }}
            
            // Helper functions
            function escapeHtml(text) {{
                const div = parentDoc.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }}
            
            function formatNumber(num) {{
                if (typeof num === 'number') {{
                    return num.toFixed(3);
                }}
                return String(num);
            }}
            
            // Initialize with retries
            let retryCount = 0;
            const maxRetries = 10;
            
            function tryAttach() {{
                const plotlyDiv = findPlotlyChart();
                if (!plotlyDiv && retryCount < maxRetries) {{
                    retryCount++;
                    console.log(`[Structure Viewer {chart_id}] Retry ${{retryCount}}/${{maxRetries}} in 500ms`);
                    setTimeout(tryAttach, 500);
                }} else if (plotlyDiv) {{
                    attachClickListener();
                }} else {{
                    console.error('[Structure Viewer {chart_id}] Failed to find Plotly chart after ${{maxRetries}} retries');
                }}
            }}
            
            tryAttach();
            
            // Re-attach listener when Streamlit updates (with debouncing for performance)
            let debounceTimer;
            const observer = new MutationObserver(function(mutations) {{
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(() => {{
                    const plotlyDiv = findPlotlyChart();
                    if (plotlyDiv && !plotlyDiv._structureViewerAttached_{chart_id}) {{
                        console.log('[Structure Viewer {chart_id}] Detected plot update, re-attaching');
                        attachClickListener();
                    }}
                }}, 100);  // 100ms debounce
            }});
            
            const container = parentDoc.querySelector('[data-testid="stAppViewContainer"]');
            if (container) {{
                observer.observe(container, {{ childList: true, subtree: true }});
                console.log('[Structure Viewer {chart_id}] ✅ Mutation observer active');
            }}
        }}
    }})();
    </script>
    """
    
    return html_component


def get_structure_viewer_hint():
    """
    Generate a hint message to display above charts that support structure viewing.
    
    Returns:
        HTML string with hint message
    """
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
        <span style="font-size: 16px;">📍</span>
        <span><strong>Tip:</strong> Click on any data point to view its 2D molecular structure in the side panel</span>
    </div>
    """
