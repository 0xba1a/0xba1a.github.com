(() => {
  const stage = document.getElementById('stage');
  const stepCounter = document.getElementById('stepCounter');
  const clickHint = document.getElementById('clickHint');

  const sceneHook     = document.querySelector('.scene-hook');
  const sceneSabotage = document.querySelector('.scene-sabotage');
  const sceneReveal   = document.querySelector('.scene-reveal');
  const sceneFF       = document.querySelector('.scene-fault-failure');
  const sceneJob      = document.querySelector('.scene-job');
  const sceneInject   = document.querySelector('.scene-inject');
  const sceneWorkflow = document.querySelector('.scene-workflow');
  const sceneTakeaway = document.querySelector('.scene-takeaway');
  const allScenes = [
    sceneHook, sceneSabotage, sceneReveal, sceneFF,
    sceneJob, sceneInject, sceneWorkflow, sceneTakeaway,
  ];

  const TOTAL_STEPS = 9;
  let currentStep = 0;
  let initedHook = false, initedSabotage = false, initedFF = false,
      initedInject = false, initedWorkflow = false;

  /* ────────── helpers ────────── */
  function showScene(el) {
    allScenes.forEach(s => s.classList.remove('active'));
    el.classList.add('active');
  }
  function shake(ms = 500) {
    stage.classList.remove('shake');
    void stage.offsetWidth;
    stage.classList.add('shake');
    setTimeout(() => stage.classList.remove('shake'), ms);
  }

  /* ════════════════════════════════════════════
     Scene 1 — Hook: rack row + cable yank
     ════════════════════════════════════════════ */
  function initHook() {
    if (initedHook) return;
    initedHook = true;
    const W = 800, H = 460;
    const svg = d3.select('#hookStage').append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    const RACKS = 6;
    const rackW = 90, rackH = 260;
    const gap = (W - RACKS * rackW) / (RACKS + 1);
    const baseY = (H - rackH) / 2;

    // Floor line
    svg.append('line')
      .attr('x1', 20).attr('x2', W - 20)
      .attr('y1', baseY + rackH + 14).attr('y2', baseY + rackH + 14)
      .attr('stroke', '#30363d').attr('stroke-width', 1);

    const rackG = svg.selectAll('g.rack')
      .data(d3.range(RACKS))
      .enter().append('g')
      .attr('class', d => `rack rack-${d}`)
      .attr('transform', (d, i) => `translate(${gap + i * (rackW + gap)},${baseY})`);

    rackG.append('rect')
      .attr('class', 'rack-body')
      .attr('width', rackW).attr('height', rackH)
      .attr('rx', 6);

    // 4 LED rows per rack
    rackG.each(function () {
      const g = d3.select(this);
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 3; c++) {
          g.append('circle')
            .attr('class', `rack-led blink rack-led-${r}-${c}`)
            .attr('cx', 18 + c * 22)
            .attr('cy', 30 + r * 50)
            .attr('r', 5)
            .style('animation-delay', `${(r * 3 + c) * 80}ms`);
        }
      }
    });

    // Cable — connects rack #3 (deterministic target) to top-right exit
    const TARGET = 3;
    const targetCx = gap + TARGET * (rackW + gap) + rackW / 2;
    const targetCy = baseY + 20;
    const cable = svg.append('path')
      .attr('class', 'cable-path')
      .attr('id', 'hookCable')
      .attr('d', `M ${targetCx} ${targetCy} C ${targetCx + 40} ${targetCy - 60}, ${W - 80} ${targetCy - 100}, ${W - 30} 30`);

    // Sparks group (revealed on yank)
    const sparkG = svg.append('g').attr('class', 'sparks').attr('transform', `translate(${targetCx},${targetCy})`);
    for (let i = 0; i < 7; i++) {
      const ang = (Math.PI * 2 * i) / 7;
      sparkG.append('line')
        .attr('class', 'spark-line')
        .attr('x1', 0).attr('y1', 0)
        .attr('x2', Math.cos(ang) * 24)
        .attr('y2', Math.sin(ang) * 24)
        .style('opacity', 0);
    }

    // Auto-yank after a beat
    setTimeout(() => {
      // Pull cable away
      cable.transition().duration(500)
        .attr('d', `M ${targetCx + 200} ${targetCy - 80} C ${targetCx + 240} ${targetCy - 140}, ${W - 60} ${targetCy - 160}, ${W - 30} 30`)
        .style('opacity', .35);

      // Sparks flash
      sparkG.selectAll('line')
        .transition().duration(120).style('opacity', 1)
        .transition().duration(280).style('opacity', 0);

      // Target rack LEDs go red and stop blinking
      svg.selectAll(`.rack-${TARGET} .rack-led`)
        .classed('blink', false)
        .classed('dead', true);
    }, 700);
  }

  /* ════════════════════════════════════════════
     Scene 2 — Three sabotage acts
     ════════════════════════════════════════════ */
  function initSabotage() {
    if (initedSabotage) return;
    initedSabotage = true;

    const W = 800, H = 1100;
    const svg = d3.select('#sabotageStage').append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    const ACTS = [
      { y:  80, label: 'Service crash',         tag: 'kill -9' },
      { y: 420, label: 'Network cable pulled',  tag: 'X' },
      { y: 760, label: 'Datacenter offline',    tag: 'POWER OFF' },
    ];

    ACTS.forEach((act, i) => {
      const g = svg.append('g')
        .attr('class', `sab-act sab-act-${i}`)
        .attr('transform', `translate(${W/2 - 320},${act.y})`)
        .style('opacity', 0);

      g.append('rect')
        .attr('class', 'sab-card-body')
        .attr('width', 640).attr('height', 280)
        .attr('rx', 18);

      // Label
      g.append('text')
        .attr('class', 'sab-label')
        .attr('x', 320).attr('y', 50)
        .attr('text-anchor', 'middle')
        .style('font-size', '24px')
        .text(act.label);

      // Per-act icon
      if (i === 0) drawMicroservice(g, 320, 160, 80);
      if (i === 1) drawCable(g, 320, 160);
      if (i === 2) drawDatacenter(g, 320, 160);

      // Tag
      g.append('text')
        .attr('class', 'sab-tag')
        .attr('x', 320).attr('y', 250)
        .attr('text-anchor', 'middle')
        .style('font-size', '20px')
        .text(act.tag);
    });

    function drawMicroservice(parent, cx, cy, r) {
      const pts = d3.range(6).map(i => {
        const a = (Math.PI * 2 * i) / 6 - Math.PI / 2;
        return [cx + Math.cos(a) * r, cy + Math.sin(a) * r];
      });
      parent.append('polygon')
        .attr('class', 'sab-icon msvc')
        .attr('points', pts.map(p => p.join(',')).join(' '));
    }
    function drawCable(parent, cx, cy) {
      // Two halves of a cable, separated
      parent.append('rect').attr('class', 'sab-icon cable')
        .attr('x', cx - 110).attr('y', cy - 14).attr('width', 80).attr('height', 28).attr('rx', 6);
      parent.append('rect').attr('class', 'sab-icon cable')
        .attr('x', cx + 30).attr('y', cy - 14).attr('width', 80).attr('height', 28).attr('rx', 6);
      parent.append('line').attr('class', 'sab-x x1')
        .attr('x1', cx - 16).attr('y1', cy - 16)
        .attr('x2', cx + 16).attr('y2', cy + 16);
      parent.append('line').attr('class', 'sab-x x2')
        .attr('x1', cx - 16).attr('y1', cy + 16)
        .attr('x2', cx + 16).attr('y2', cy - 16);
    }
    function drawDatacenter(parent, cx, cy) {
      const w = 240, h = 130;
      parent.append('rect').attr('class', 'sab-icon dc-roof')
        .attr('x', cx - w/2).attr('y', cy - h/2).attr('width', w).attr('height', h).attr('rx', 6);
      // 5×4 windows
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 6; c++) {
          parent.append('rect').attr('class', `sab-window dc-w-${r}-${c}`)
            .attr('x', cx - w/2 + 14 + c * 36)
            .attr('y', cy - h/2 + 10 + r * 28)
            .attr('width', 24).attr('height', 18).attr('rx', 2)
            .attr('fill', '#3fb950');
        }
      }
    }
  }

  function runSabotage() {
    const counterEl = document.getElementById('sabotageCounter');
    counterEl.textContent = 'Faults injected: 0';

    const acts = d3.selectAll('.sab-act');
    acts.each(function(_, i) {
      const node = d3.select(this);
      setTimeout(() => {
        node.transition().duration(300)
          .style('opacity', 1)
          .style('transform', `translate(${800/2 - 320}px, ${[80,420,760][i]}px) scale(1)`);
        // Kill effect: card body and icon turn red after a beat
        setTimeout(() => {
          node.select('rect.sab-card-body').classed('killed', true);
          node.selectAll('.sab-icon').classed('killed', true);
          if (i === 2) {
            // Datacenter lights out — sweep
            node.selectAll('.sab-window').each(function(_, k) {
              const win = d3.select(this);
              setTimeout(() => win.attr('fill', '#21262d'), k * 35);
            });
          }
          counterEl.textContent = `Faults injected: ${i + 1}`;
        }, 350);
      }, i * 1300);
    });
  }

  /* ════════════════════════════════════════════
     Scene 4 — Fault vs Failure rings
     ════════════════════════════════════════════ */
  function buildRing(containerId, kind /* 'fault' | 'failure' */) {
    const W = 600, H = 380;
    const svg = d3.select(containerId).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    const N = 5;
    const cx = W / 2, cy = H / 2;
    const R = 130;
    const nodes = d3.range(N).map(i => {
      const a = (Math.PI * 2 * i) / N - Math.PI / 2;
      return { i, x: cx + Math.cos(a) * R, y: cy + Math.sin(a) * R };
    });
    const edges = [];
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) edges.push({ a: i, b: j });
    }

    svg.append('g').selectAll('line.node-edge')
      .data(edges).enter().append('line')
      .attr('class', d => `node-edge edge-${kind}-${d.a}-${d.b}`)
      .attr('x1', d => nodes[d.a].x).attr('y1', d => nodes[d.a].y)
      .attr('x2', d => nodes[d.b].x).attr('y2', d => nodes[d.b].y);

    svg.append('g').selectAll('circle.node-circle')
      .data(nodes).enter().append('circle')
      .attr('class', d => `node-circle node-${kind}-${d.i}`)
      .attr('cx', d => d.x).attr('cy', d => d.y)
      .attr('r', 26);

    return { svg, nodes, edges };
  }

  function initFF() {
    if (initedFF) return;
    initedFF = true;
    buildRing('#faultStage', 'fault');
    buildRing('#failureStage', 'failure');
  }

  function runFault() {
    // FAULT side only: kill node 0, sever its edges, surviving network thickens
    const N = 5;
    const dead = 0;
    setTimeout(() => {
      d3.select(`.node-fault-${dead}`)
        .classed('dead', true)
        .transition().duration(500)
        .attr('r', 14);
      // Sever all edges touching the dead node
      d3.selectAll('line.node-edge')
        .filter(function() {
          const cls = this.getAttribute('class');
          return cls.includes(`edge-fault-${dead}-`) || cls.endsWith(`-${dead}`) && cls.includes('fault');
        })
        .classed('severed', true);
      // Surviving edges (those not touching dead) → bypass blue
      d3.selectAll('line.node-edge')
        .filter(function() {
          const cls = this.getAttribute('class');
          if (!cls.includes('fault')) return false;
          if (cls.includes('severed')) return false;
          return true;
        })
        .transition().delay(300).duration(400)
        .each(function() { d3.select(this).classed('bypass', true); });
      // Surviving nodes pulse green
      for (let i = 0; i < N; i++) {
        if (i === dead) continue;
        d3.select(`.node-fault-${i}`)
          .transition().delay(600).duration(220).attr('r', 30)
          .transition().duration(220).attr('r', 26);
      }
    }, 200);
  }

  function runFailure() {
    // FAILURE side: ONE node turns red first, then the cascade
    sceneFF.classList.add('show-failure');

    const order = [0, 1, 4, 2, 3]; // BFS-ish around the ring
    const FIRST = order[0];
    const FIRST_DELAY = 700;       // how long the first red node sits alone
    const STEP = 180;              // delay between subsequent reds

    setTimeout(() => {
      // Patient zero — one node, alone, turns red
      d3.select(`.node-failure-${FIRST}`).classed('dead', true);
    }, 350);

    // After the beat, cascade through the rest
    order.slice(1).forEach((i, k) => {
      setTimeout(() => {
        d3.select(`.node-failure-${i}`).classed('dead', true);
        // Color edges between i and any already-dead node
        const deadSoFar = order.slice(0, k + 2);
        deadSoFar.forEach(j => {
          if (j === i) return;
          const a = Math.min(i, j), b = Math.max(i, j);
          d3.select(`.edge-failure-${a}-${b}`).classed('dead', true);
        });
        if (k === order.length - 2) shake(420);
      }, 350 + FIRST_DELAY + k * STEP);
    });
  }

  /* ════════════════════════════════════════════
     Scene 6 — Inject the fault
     ════════════════════════════════════════════ */
  let injectSvg = null;

  function initInject() {
    if (initedInject) return;
    initedInject = true;
    const W = 800, H = 1000;
    injectSvg = d3.select('#injectStage').append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    // 3×3 grid of services
    const COLS = 3, ROWS = 3;
    const padX = 100, padY = 130;
    const cw = (W - padX * 2) / (COLS - 1);
    const ch = (H - padY * 2) / (ROWS - 1);

    const nodes = [];
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        nodes.push({ i: r * COLS + c, r, c, x: padX + c * cw, y: padY + r * ch });
      }
    }

    // Edges: 4-neighbor connectivity
    const edges = [];
    nodes.forEach(n => {
      nodes.forEach(m => {
        if (m.i <= n.i) return;
        if ((Math.abs(n.r - m.r) === 1 && n.c === m.c) ||
            (Math.abs(n.c - m.c) === 1 && n.r === m.r)) {
          edges.push({ a: n.i, b: m.i });
        }
      });
    });

    injectSvg.append('g').selectAll('line.svc-edge')
      .data(edges).enter().append('line')
      .attr('class', d => `svc-edge svc-edge-${d.a}-${d.b}`)
      .attr('x1', d => nodes[d.a].x).attr('y1', d => nodes[d.a].y)
      .attr('x2', d => nodes[d.b].x).attr('y2', d => nodes[d.b].y);

    injectSvg.append('g').selectAll('g.svc')
      .data(nodes).enter().append('g')
      .attr('class', d => `svc svc-${d.i}`)
      .attr('transform', d => `translate(${d.x},${d.y})`)
      .each(function() {
        const g = d3.select(this);
        g.append('rect')
          .attr('class', 'svc-body healthy')
          .attr('x', -52).attr('y', -36)
          .attr('width', 104).attr('height', 72)
          .attr('rx', 10);
        g.append('text')
          .attr('class', 'svc-label')
          .attr('text-anchor', 'middle')
          .attr('y', 6)
          .text(d => `svc-${d.i}`);
      });

    // Syringe (parked above)
    const syrG = injectSvg.append('g')
      .attr('class', 'syringe-g')
      .attr('id', 'syringeG')
      .attr('transform', 'translate(700,80)');
    syrG.append('rect')
      .attr('class', 'syringe-body')
      .attr('x', -50).attr('y', -16).attr('width', 90).attr('height', 32).attr('rx', 6);
    syrG.append('line')
      .attr('class', 'syringe-tip')
      .attr('x1', 40).attr('y1', 0).attr('x2', 78).attr('y2', 0);
    syrG.append('text')
      .attr('class', 'syringe-label')
      .attr('x', -8).attr('y', -28).attr('text-anchor', 'end')
      .text('inject()');

    injectSvg.datum({ nodes, edges });
  }

  function runInject() {
    const meterEl = document.getElementById('injectMeter');
    meterEl.textContent = '— ms';
    meterEl.style.color = '#3fb950';

    const { nodes, edges } = injectSvg.datum();
    const TARGET = 4; // center

    // Reset state
    injectSvg.selectAll('.svc-body').attr('class', 'svc-body healthy');
    injectSvg.selectAll('.svc-edge').attr('class', function() {
      const m = this.getAttribute('class').match(/svc-edge-(\d+)-(\d+)/);
      return `svc-edge svc-edge-${m[1]}-${m[2]}`;
    });
    injectSvg.selectAll('.breaker-line').remove();

    const target = nodes[TARGET];

    // Move syringe to target
    const syr = injectSvg.select('#syringeG');
    syr.style('transform', `translate(${target.x + 70}px, ${target.y - 70}px)`);

    setTimeout(() => {
      // Inject ink — red ripple at target node
      injectSvg.append('circle')
        .attr('cx', target.x).attr('cy', target.y).attr('r', 30)
        .attr('fill', '#f85149').style('opacity', .6)
        .transition().duration(700)
        .attr('r', 130).style('opacity', 0)
        .remove();

      // Target turns red
      injectSvg.select(`.svc-${TARGET} .svc-body`)
        .attr('class', 'svc-body dead');

      // Neighbors flicker stressed (orange) then back to healthy
      const neighbors = edges
        .filter(e => e.a === TARGET || e.b === TARGET)
        .map(e => e.a === TARGET ? e.b : e.a);
      neighbors.forEach(ni => {
        const sel = injectSvg.select(`.svc-${ni} .svc-body`);
        sel.attr('class', 'svc-body stressed');
        setTimeout(() => sel.attr('class', 'svc-body healthy'), 700);
      });

      // Sever target edges — circuit breaker tick marks
      const targetEdges = edges
        .filter(e => e.a === TARGET || e.b === TARGET);
      targetEdges.forEach(e => {
        injectSvg.select(`.svc-edge-${Math.min(e.a, e.b)}-${Math.max(e.a, e.b)}`)
          .classed('severed', true);
        // Draw a small breaker mark at the midpoint
        const a = nodes[e.a], b = nodes[e.b];
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        const dx = b.x - a.x, dy = b.y - a.y;
        const len = Math.hypot(dx, dy);
        const nx = -dy / len, ny = dx / len;
        injectSvg.append('line')
          .attr('class', 'breaker-line')
          .attr('x1', mx + nx * 10).attr('y1', my + ny * 10)
          .attr('x2', mx - nx * 10).attr('y2', my - ny * 10)
          .style('opacity', 0)
          .transition().delay(300).duration(250).style('opacity', 1);
      });

      // Recovery counter ticks 0 → 187ms
      const t0 = Date.now();
      const FINAL = 187;
      const tick = () => {
        const dt = Math.min((Date.now() - t0) / 1200, 1);
        const v = Math.floor(d3.easeQuadOut(dt) * FINAL);
        meterEl.textContent = `${v} ms`;
        if (dt < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }, 850);
  }

  /* ════════════════════════════════════════════
     Scene 7 — Workflow loop
     ════════════════════════════════════════════ */
  function initWorkflow() {
    if (initedWorkflow) return;
    initedWorkflow = true;
    const W = 800, H = 700;
    const svg = d3.select('#workflowStage').append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    // arrowhead
    svg.append('defs').append('marker')
      .attr('id', 'wfArrow').attr('viewBox', '0 0 10 10')
      .attr('refX', 9).attr('refY', 5)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M0,0 L10,5 L0,10 z').attr('fill', '#58a6ff');
    svg.append('defs').append('marker')
      .attr('id', 'wfLoopArrow').attr('viewBox', '0 0 10 10')
      .attr('refX', 9).attr('refY', 5)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M0,0 L10,5 L0,10 z').attr('fill', '#d2a8ff');

    const BLOCKS = [
      { label: 'Inject',  x: 130, y: 360 },
      { label: 'Observe', x: 400, y: 200 },
      { label: 'Fix',     x: 670, y: 360 },
    ];
    const BW = 180, BH = 90;

    // Connecting arrows (drawn first so blocks paint over)
    const arrowsG = svg.append('g').attr('class', 'wf-arrows');
    arrowsG.append('path').attr('class', 'wf-arrow a-1')
      .attr('d', `M ${BLOCKS[0].x + BW/2 - 4} ${BLOCKS[0].y} C ${(BLOCKS[0].x + BLOCKS[1].x)/2} ${BLOCKS[0].y - 60}, ${(BLOCKS[0].x + BLOCKS[1].x)/2} ${BLOCKS[1].y + 80}, ${BLOCKS[1].x - BW/2 + 4} ${BLOCKS[1].y + 8}`)
      .attr('marker-end', 'url(#wfArrow)')
      .attr('stroke-dasharray', 600).attr('stroke-dashoffset', 600);
    arrowsG.append('path').attr('class', 'wf-arrow a-2')
      .attr('d', `M ${BLOCKS[1].x + BW/2 - 4} ${BLOCKS[1].y + 8} C ${(BLOCKS[1].x + BLOCKS[2].x)/2} ${BLOCKS[1].y + 80}, ${(BLOCKS[1].x + BLOCKS[2].x)/2} ${BLOCKS[2].y - 60}, ${BLOCKS[2].x - BW/2 + 4} ${BLOCKS[2].y}`)
      .attr('marker-end', 'url(#wfArrow)')
      .attr('stroke-dasharray', 600).attr('stroke-dashoffset', 600);

    // Loop-back arrow (Fix → Inject)
    arrowsG.append('path').attr('class', 'wf-loop')
      .attr('d', `M ${BLOCKS[2].x} ${BLOCKS[2].y + BH/2 - 4} C ${BLOCKS[2].x} ${BLOCKS[2].y + 220}, ${BLOCKS[0].x} ${BLOCKS[0].y + 220}, ${BLOCKS[0].x} ${BLOCKS[0].y + BH/2 - 4}`)
      .attr('marker-end', 'url(#wfLoopArrow)')
      .attr('stroke-dasharray', 1000).attr('stroke-dashoffset', 1000);
    arrowsG.append('text').attr('class', 'wf-loop-label')
      .attr('x', (BLOCKS[0].x + BLOCKS[2].x) / 2).attr('y', BLOCKS[2].y + 250)
      .attr('text-anchor', 'middle').text('repeat')
      .style('opacity', 0);

    // Blocks
    const blockG = svg.selectAll('g.wf-block')
      .data(BLOCKS).enter().append('g')
      .attr('class', (d, i) => `wf-block wf-block-${i}`)
      .attr('transform', d => `translate(${d.x - BW/2},${d.y - BH/2})`)
      .style('opacity', 0);
    blockG.append('rect')
      .attr('class', 'wf-block-body')
      .attr('width', BW).attr('height', BH).attr('rx', 12);
    blockG.append('text')
      .attr('class', 'wf-block-label')
      .attr('x', BW/2).attr('y', BH/2 + 6)
      .attr('text-anchor', 'middle')
      .text(d => d.label);
  }

  function runWorkflow() {
    // Stagger blocks → arrows → loop
    d3.selectAll('.wf-block').each(function(_, i) {
      d3.select(this).transition().delay(i * 350).duration(300).style('opacity', 1);
    });
    d3.select('.a-1').transition().delay(420).duration(500).attr('stroke-dashoffset', 0);
    d3.select('.a-2').transition().delay(720).duration(500).attr('stroke-dashoffset', 0);
    d3.select('.wf-loop').transition().delay(1100).duration(900).attr('stroke-dashoffset', 0);
    d3.select('.wf-loop-label').transition().delay(1900).duration(400).style('opacity', 1);
  }

  /* ════════════════════════════════════════════
     Step machine
     ════════════════════════════════════════════ */
  function step1() {
    showScene(sceneHook);
    initHook();
  }
  function step2() {
    showScene(sceneSabotage);
    initSabotage();
    setTimeout(runSabotage, 250);
  }
  function step3() { showScene(sceneReveal); }
  function step4() {
    // Fault only — top half of the split scene
    showScene(sceneFF);
    sceneFF.classList.remove('show-failure');
    initFF();
    setTimeout(runFault, 250);
  }
  function step5() {
    // Reveal failure portion + run the cascade
    runFailure();
  }
  function step6() { showScene(sceneJob); }
  function step7() {
    showScene(sceneInject);
    initInject();
    setTimeout(runInject, 250);
  }
  function step8() {
    showScene(sceneWorkflow);
    initWorkflow();
    setTimeout(runWorkflow, 200);
  }
  function step9() { showScene(sceneTakeaway); }

  const steps = [null, step1, step2, step3, step4, step5, step6, step7, step8, step9];

  function updateCounter() { stepCounter.textContent = `${currentStep} / ${TOTAL_STEPS}`; }

  function advance() {
    if (currentStep >= TOTAL_STEPS) return;
    currentStep++;
    steps[currentStep]();
    updateCounter();
    if (currentStep >= TOTAL_STEPS) clickHint.textContent = '← Tap to restart';
  }

  function reset() {
    currentStep = 0;
    allScenes.forEach(s => s.classList.remove('active'));
    sceneFF.classList.remove('show-failure');

    // Reset Scene 4 rings
    d3.selectAll('.node-circle').classed('dead', false).classed('fading', false).attr('r', 26);
    d3.selectAll('.node-edge').classed('dead', false).classed('severed', false).classed('bypass', false);

    // Reset Scene 6 inject
    if (injectSvg) {
      injectSvg.selectAll('.svc-body').attr('class', 'svc-body healthy');
      injectSvg.selectAll('.svc-edge').each(function() {
        const m = this.getAttribute('class').match(/svc-edge-(\d+)-(\d+)/);
        if (m) this.setAttribute('class', `svc-edge svc-edge-${m[1]}-${m[2]}`);
      });
      injectSvg.selectAll('.breaker-line').remove();
      injectSvg.select('#syringeG').style('transform', 'translate(700px,80px)');
    }
    document.getElementById('injectMeter').textContent = '— ms';
    document.getElementById('sabotageCounter').textContent = 'Faults injected: 0';

    // Reset Scene 2 acts
    d3.selectAll('.sab-act').style('opacity', 0);
    d3.selectAll('.sab-card-body').classed('killed', false);
    d3.selectAll('.sab-icon').classed('killed', false);
    d3.selectAll('.sab-window').attr('fill', '#3fb950');

    // Reset Scene 7 workflow
    d3.selectAll('.wf-block').style('opacity', 0);
    d3.selectAll('.wf-arrow').attr('stroke-dashoffset', 600);
    d3.select('.wf-loop').attr('stroke-dashoffset', 1000);
    d3.select('.wf-loop-label').style('opacity', 0);

    clickHint.textContent = 'Tap or press → to advance';
    updateCounter();
  }

  /* ── Debounce ── */
  let lastAdvance = 0;
  const DEBOUNCE_MS = 300;
  function handleAdvance() {
    const now = Date.now();
    if (now - lastAdvance < DEBOUNCE_MS) return;
    lastAdvance = now;
    if (currentStep >= TOTAL_STEPS) reset();
    else advance();
  }

  stage.addEventListener('click', handleAdvance);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === ' ') {
      e.preventDefault();
      handleAdvance();
    }
    if (e.key === 'ArrowLeft' || e.key === 'r') reset();
  });

  updateCounter();
})();
