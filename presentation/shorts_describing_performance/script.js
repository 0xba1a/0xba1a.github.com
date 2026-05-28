(() => {
  // ─────────────────────────────────────────
  // State
  // ─────────────────────────────────────────
  const TOTAL_STEPS = 8;
  let currentStep = 0;
  let lastClickTime = 0;
  const CLICK_DEBOUNCE = 350;

  const stage = document.getElementById('stage');
  const stepCounter = document.getElementById('stepCounter');
  const clickHint = document.getElementById('clickHint');
  const chartTitle = document.getElementById('chartTitle');
  const chartContainer = document.getElementById('chartContainer');

  const scenes = {
    hook:      document.querySelector('.scene-hook'),
    metrics:   document.querySelector('.scene-metrics'),
    chart:     document.querySelector('.scene-chart'),
    statement: document.querySelector('.scene-statement'),
  };

  // ─────────────────────────────────────────
  // Chart constants
  // ─────────────────────────────────────────
  const W = 600, H = 600;
  const M = { top: 30, right: 40, bottom: 85, left: 90 };
  const iw = W - M.left - M.right;
  const ih = H - M.top - M.bottom;

  // Queuing model
  const CAPACITY = 1.2;   // 1.2M req/s
  const RT0 = 3.5;        // baseline ms
  const RT_MAX_DOMAIN = 200;
  const rt = (lambda) => Math.min(RT_MAX_DOMAIN * 1.1, RT0 / Math.max(0.01, 1 - lambda / CAPACITY));

  // Key points
  const SWEET = { x: 1.0,  y: rt(1.0) };   // ~21ms
  const DANGER = { x: 1.15, y: rt(1.15) }; // huge
  const WASTE  = { x: 0.25, y: rt(0.25) }; // ~4.4ms

  let svg = null;
  let xScale, yScale, lineGen, curvePath;
  let chartInitialized = false;

  // ─────────────────────────────────────────
  // Chart init (lazy, runs on step 3)
  // ─────────────────────────────────────────
  function initChart() {
    if (chartInitialized) return;
    chartInitialized = true;

    chartContainer.innerHTML = '';
    svg = d3.select(chartContainer).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    const g = svg.append('g')
      .attr('transform', `translate(${M.left}, ${M.top})`)
      .attr('id', 'plot');

    xScale = d3.scaleLinear().domain([0, CAPACITY]).range([0, iw]);
    yScale = d3.scaleLinear().domain([0, RT_MAX_DOMAIN]).range([ih, 0]);

    // Axes (initially hidden)
    const xAxis = d3.axisBottom(xScale)
      .tickValues([0, 0.3, 0.6, 0.9, 1.2])
      .tickFormat(d => d === 0 ? '0' : `${d.toFixed(1)}M`);
    const yAxis = d3.axisLeft(yScale)
      .tickValues([0, 50, 100, 150, 200])
      .tickFormat(d => d === 0 ? '0' : `${d}ms`);

    g.append('g')
      .attr('class', 'axis x-axis')
      .attr('transform', `translate(0, ${ih})`)
      .attr('opacity', 0)
      .call(xAxis);

    g.append('g')
      .attr('class', 'axis y-axis')
      .attr('opacity', 0)
      .call(yAxis);

    // Axis titles
    g.append('text')
      .attr('class', 'axis-title x-title')
      .attr('x', iw / 2)
      .attr('y', ih + 64)
      .attr('text-anchor', 'middle')
      .attr('opacity', 0)
      .text('Throughput (req/s)');

    g.append('text')
      .attr('class', 'axis-title y-title')
      .attr('transform', `translate(-66, ${ih / 2}) rotate(-90)`)
      .attr('text-anchor', 'middle')
      .attr('opacity', 0)
      .text('Response Time');

    // Curve path (hidden, populated on step 4)
    lineGen = d3.line()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y))
      .curve(d3.curveMonotoneX);

    const points = [];
    for (let i = 0; i <= 120; i++) {
      const lam = (CAPACITY * 1.0) * (i / 120);
      points.push({ x: lam, y: rt(lam) });
    }

    curvePath = g.append('path')
      .datum(points)
      .attr('class', 'curve')
      .attr('d', lineGen);

    const totalLen = curvePath.node().getTotalLength();
    curvePath
      .attr('stroke-dasharray', `${totalLen} ${totalLen}`)
      .attr('stroke-dashoffset', totalLen);
  }

  // ─────────────────────────────────────────
  // Helpers
  // ─────────────────────────────────────────
  function setActiveScene(name) {
    Object.entries(scenes).forEach(([k, el]) => {
      el.classList.toggle('active', k === name);
    });
  }

  function clearChartOverlays() {
    if (!svg) return;
    svg.select('#plot').selectAll('.overlay').remove();
  }

  function plotG() { return svg.select('#plot'); }

  function showAxes() {
    plotG().selectAll('.axis, .axis-title')
      .transition().duration(500).attr('opacity', 1);
  }

  function drawCurve() {
    const totalLen = curvePath.node().getTotalLength();
    curvePath
      .attr('stroke-dashoffset', totalLen)
      .transition().duration(1200).ease(d3.easeCubicOut)
      .attr('stroke-dashoffset', 0);
  }

  function showDangerMarker() {
    const g = plotG();
    // Marker starts at sweet spot and moves to danger
    const m = g.append('circle')
      .attr('class', 'overlay marker marker-danger')
      .attr('cx', xScale(SWEET.x))
      .attr('cy', yScale(SWEET.y))
      .attr('r', 0);

    m.transition().duration(250).attr('r', 10)
     .transition().duration(900).ease(d3.easeCubicIn)
       .attr('cx', xScale(DANGER.x))
       .attr('cy', yScale(DANGER.y));

    // Label
    g.append('text')
      .attr('class', 'overlay zone-label zone-danger')
      .attr('x', xScale(DANGER.x) - 12)
      .attr('y', yScale(Math.min(DANGER.y, RT_MAX_DOMAIN * 0.85)) - 16)
      .attr('text-anchor', 'end')
      .attr('opacity', 0)
      .text('💥 Queues explode')
      .transition().delay(1100).duration(400).attr('opacity', 1);

    g.append('text')
      .attr('class', 'overlay callout')
      .attr('x', xScale(DANGER.x) - 12)
      .attr('y', yScale(Math.min(DANGER.y, RT_MAX_DOMAIN * 0.85)) + 8)
      .attr('text-anchor', 'end')
      .attr('opacity', 0)
      .text('RT → ∞')
      .transition().delay(1300).duration(400).attr('opacity', 1);
  }

  function showWasteMarker() {
    clearChartOverlays();
    const g = plotG();
    // Start at danger position visually (jump from prev), move to waste
    const m = g.append('circle')
      .attr('class', 'overlay marker marker-waste')
      .attr('cx', xScale(DANGER.x))
      .attr('cy', yScale(Math.min(DANGER.y, RT_MAX_DOMAIN)))
      .attr('r', 10);

    m.transition().duration(1000).ease(d3.easeCubicInOut)
       .attr('cx', xScale(WASTE.x))
       .attr('cy', yScale(WASTE.y));

    g.append('text')
      .attr('class', 'overlay zone-label zone-waste')
      .attr('x', xScale(WASTE.x) + 18)
      .attr('y', yScale(WASTE.y) - 32)
      .attr('opacity', 0)
      .text('Wasted capacity')
      .transition().delay(900).duration(400).attr('opacity', 1);

    g.append('text')
      .attr('class', 'overlay callout')
      .attr('x', xScale(WASTE.x) + 18)
      .attr('y', yScale(WASTE.y) - 12)
      .attr('opacity', 0)
      .text('Low throughput')
      .transition().delay(1100).duration(400).attr('opacity', 1);
  }

  function showSweetSpot() {
    clearChartOverlays();
    const g = plotG();
    const sx = xScale(SWEET.x);
    const sy = yScale(SWEET.y);

    // Crosshairs
    g.append('line')
      .attr('class', 'overlay crosshair')
      .attr('x1', sx).attr('x2', sx)
      .attr('y1', ih).attr('y2', ih)
      .transition().duration(600).attr('y2', sy);

    g.append('line')
      .attr('class', 'overlay crosshair')
      .attr('x1', 0).attr('x2', 0)
      .attr('y1', sy).attr('y2', sy)
      .transition().duration(600).attr('x2', sx);

    // Axis-edge labels
    g.append('text')
      .attr('class', 'overlay crosshair-label')
      .attr('x', sx + 6)
      .attr('y', ih - 8)
      .attr('text-anchor', 'start')
      .attr('opacity', 0)
      .text('1M req/s')
      .transition().delay(600).duration(400).attr('opacity', 1);

    g.append('text')
      .attr('class', 'overlay crosshair-label')
      .attr('x', -12)
      .attr('y', sy + 6)
      .attr('text-anchor', 'end')
      .attr('opacity', 0)
      .text('20ms')
      .transition().delay(600).duration(400).attr('opacity', 1);

    // Sweet spot dot with pulse
    const dot = g.append('circle')
      .attr('class', 'overlay marker marker-sweet')
      .attr('cx', sx).attr('cy', sy)
      .attr('r', 0);

    dot.transition().delay(400).duration(400).ease(d3.easeBackOut.overshoot(2))
      .attr('r', 12);

    // Pulse ring
    function pulse() {
      g.append('circle')
        .attr('class', 'overlay')
        .attr('cx', sx).attr('cy', sy)
        .attr('r', 12)
        .attr('fill', 'none')
        .attr('stroke', '#3fb950')
        .attr('stroke-width', 2)
        .attr('opacity', 0.8)
        .transition().duration(1400).ease(d3.easeCubicOut)
          .attr('r', 32)
          .attr('opacity', 0)
        .remove();
    }
    pulse();
    const pulseInterval = setInterval(pulse, 1100);
    // Stop pulsing if we move away
    g.attr('data-pulse-id', pulseInterval);

    // Sweet spot label
    g.append('text')
      .attr('class', 'overlay zone-label zone-sweet')
      .attr('x', sx - 22)
      .attr('y', sy - 22)
      .attr('text-anchor', 'end')
      .attr('opacity', 0)
      .text('Sweet spot')
      .transition().delay(900).duration(400).attr('opacity', 1);

    g.append('text')
      .attr('class', 'overlay callout')
      .attr('x', sx - 22)
      .attr('y', sy - 2)
      .attr('text-anchor', 'end')
      .attr('opacity', 0)
      .text('The knee')
      .transition().delay(1200).duration(400).attr('opacity', 1);
  }

  function stopPulse() {
    if (!svg) return;
    const id = plotG().attr('data-pulse-id');
    if (id) {
      clearInterval(+id);
      plotG().attr('data-pulse-id', null);
    }
  }

  // ─────────────────────────────────────────
  // Step handlers
  // ─────────────────────────────────────────
  const stepHandlers = {
    1: () => {
      stopPulse();
      setActiveScene('hook');
    },
    2: () => {
      stopPulse();
      setActiveScene('metrics');
    },
    3: () => {
      stopPulse();
      setActiveScene('chart');
      chartTitle.classList.add('visible');
      initChart();
      // Reset curve
      if (curvePath) {
        const len = curvePath.node().getTotalLength();
        curvePath.attr('stroke-dashoffset', len);
      }
      clearChartOverlays();
      showAxes();
    },
    4: () => {
      stopPulse();
      setActiveScene('chart');
      chartTitle.classList.add('visible');
      initChart();
      showAxes();
      clearChartOverlays();
      drawCurve();
    },
    5: () => {
      stopPulse();
      setActiveScene('chart');
      initChart();
      clearChartOverlays();
      showDangerMarker();
    },
    6: () => {
      stopPulse();
      setActiveScene('chart');
      initChart();
      showWasteMarker();
    },
    7: () => {
      setActiveScene('chart');
      initChart();
      showSweetSpot();
    },
    8: () => {
      stopPulse();
      setActiveScene('statement');
    },
  };

  // ─────────────────────────────────────────
  // Navigation
  // ─────────────────────────────────────────
  function goToStep(n) {
    if (n < 1 || n > TOTAL_STEPS) return;
    currentStep = n;
    stepCounter.textContent = `${currentStep} / ${TOTAL_STEPS}`;
    clickHint.textContent = currentStep === TOTAL_STEPS
      ? '← Tap to restart'
      : 'Tap or press → to advance';
    stepHandlers[n]();
  }

  function reset() {
    stopPulse();
    chartInitialized = false;
    if (chartContainer) chartContainer.innerHTML = '';
    chartTitle.classList.remove('visible');
    goToStep(1);
  }

  function advance() {
    const now = Date.now();
    if (now - lastClickTime < CLICK_DEBOUNCE) return;
    lastClickTime = now;
    if (currentStep >= TOTAL_STEPS) {
      reset();
    } else {
      goToStep(currentStep + 1);
    }
  }

  // ─────────────────────────────────────────
  // Events
  // ─────────────────────────────────────────
  stage.addEventListener('click', advance);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === ' ') {
      e.preventDefault();
      advance();
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      if (currentStep > 1) goToStep(currentStep - 1);
    } else if (e.key === 'r' || e.key === 'R') {
      reset();
    }
  });

  // ─────────────────────────────────────────
  // Init
  // ─────────────────────────────────────────
  goToStep(1);
})();
