(() => {
  const stage = document.getElementById('stage');
  const stepCounter = document.getElementById('stepCounter');
  const clickHint = document.getElementById('clickHint');
  const chartTitle = document.getElementById('chartTitle');

  const sceneHook = document.querySelector('.scene-hook');
  const sceneChart = document.querySelector('.scene-chart');
  const sceneCompare = document.querySelector('.scene-compare');
  const sceneTakeaway = document.querySelector('.scene-takeaway');
  const allScenes = [sceneHook, sceneChart, sceneCompare, sceneTakeaway];

  const TOTAL_STEPS = 8;
  let currentStep = 0;

  /* ── Hardcoded data (log-normal bulk + outliers) ── */
  const rawData = [
    22, 25, 28, 30, 31, 33, 35, 36, 38, 40,
    42, 44, 45, 47, 50, 52, 55, 58, 62, 65,
    70, 75, 82, 90, 110, 140,
    2000, 4500, 7000, 10000
  ];
  const OUTLIER_THRESHOLD = 500;

  // Deterministic shuffle using a fixed seed for reproducibility
  function seededShuffle(arr) {
    const result = [...arr];
    let seed = 42;
    for (let i = result.length - 1; i > 0; i--) {
      seed = (seed * 16807 + 0) % 2147483647;
      const j = seed % (i + 1);
      [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
  }

  const shuffledData = seededShuffle(rawData);

  // Original indices for object constancy
  let data = shuffledData.map((v, i) => ({ value: v, idx: i }));
  let sorted = false;

  /* ── D3 Setup ── */
  const container = document.getElementById('chartContainer');
  let svg, chartG, xScale, yScale, margin, width, height;

  function initChart() {
    const rect = container.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;

    margin = { top: 20, right: 16, bottom: 36, left: 52 };
    width = W - margin.left - margin.right;
    height = H - margin.top - margin.bottom;

    svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    // Arrow marker for annotations
    svg.append('defs').append('marker')
      .attr('id', 'arrowMarker')
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 10).attr('refY', 5)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M 0 0 L 10 5 L 0 10 z')
      .attr('fill', '#8b949e');

    chartG = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    xScale = d3.scaleBand()
      .domain(data.map(d => d.idx))
      .range([0, width])
      .padding(0.15);

    yScale = d3.scaleSqrt()
      .domain([0, d3.max(data, d => d.value) * 1.05])
      .range([height, 0]);

    // Y-axis
    chartG.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(yScale)
        .ticks(5)
        .tickFormat(d => d >= 1000 ? (d/1000) + 's' : d + 'ms')
        .tickSize(-width)
      );

    // X-axis label
    chartG.append('text')
      .attr('class', 'axis-label x-axis-label')
      .attr('x', width / 2)
      .attr('y', height + 30)
      .attr('text-anchor', 'middle')
      .text('Requests');

    // Y-axis label
    chartG.append('text')
      .attr('class', 'axis-label')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -40)
      .attr('text-anchor', 'middle')
      .text('Response Time');
  }

  /* ── Step 2: Draw bars with staggered animation ── */
  function drawBars() {
    chartG.selectAll('.bar')
      .data(data, d => d.idx)
      .enter()
      .append('rect')
      .attr('class', d => d.value >= OUTLIER_THRESHOLD ? 'bar bar-outlier' : 'bar bar-normal')
      .attr('x', d => xScale(d.idx))
      .attr('width', xScale.bandwidth())
      .attr('y', height)
      .attr('height', 0)
      .attr('rx', 2)
      .transition()
      .delay((d, i) => i * 30)
      .duration(500)
      .attr('y', d => yScale(d.value))
      .attr('height', d => height - yScale(d.value));
  }

  /* ── Step 3: Show average line ── */
  function showAverage() {
    const avg = d3.mean(data, d => d.value);
    const avgY = yScale(avg);

    // Average line
    chartG.append('line')
      .attr('class', 'avg-line')
      .attr('x1', 0).attr('y1', avgY)
      .attr('x2', 0).attr('y2', avgY)
      .transition().duration(700)
      .attr('x2', width);

    // Average label
    chartG.append('text')
      .attr('class', 'avg-label')
      .attr('x', width - 4)
      .attr('y', avgY - 8)
      .attr('text-anchor', 'end')
      .attr('opacity', 0)
      .text(`Avg: ${Math.round(avg)}ms`)
      .transition().delay(500).duration(400)
      .attr('opacity', 1);

    // Annotation: find the tallest visible bar and annotate near it
    chartG.append('text')
      .attr('class', 'annotation-text outlier-annot')
      .attr('x', width / 2)
      .attr('y', avgY + 20)
      .attr('text-anchor', 'middle')
      .attr('opacity', 0)
      .text('← Outliers drag it up')
      .transition().delay(800).duration(400)
      .attr('opacity', 1);
  }

  /* ── Step 4: Sort bars + show p50 ── */
  function sortBarsAndShowP50() {
    sorted = true;
    const sortedData = [...data].sort((a, b) => a.value - b.value);

    // Update domain to sorted order
    xScale.domain(sortedData.map(d => d.idx));

    // Animate bars to new positions
    chartG.selectAll('.bar')
      .transition().duration(800)
      .attr('x', d => xScale(d.idx));

    // Fade out average line & annotations
    chartG.selectAll('.avg-line, .avg-label, .outlier-annot')
      .transition().duration(400)
      .attr('opacity', 0)
      .remove();

    // Update x-axis label
    chartG.select('.x-axis-label').text('Requests (sorted)');

    // Store sorted order for later
    data._sorted = sortedData;

    // Show p50 after sort completes
    const medianIdx = Math.floor(sortedData.length / 2);
    const medianVal = sortedData[medianIdx].value;
    const medianX = xScale(sortedData[medianIdx].idx) + xScale.bandwidth() / 2;

    // Vertical p50 line (delayed to appear after sort)
    chartG.append('line')
      .attr('class', 'p50-line')
      .attr('x1', medianX).attr('x2', medianX)
      .attr('y1', height)
      .attr('y2', height)
      .transition().delay(900).duration(500)
      .attr('y2', 0);

    // p50 label
    chartG.append('text')
      .attr('class', 'p50-label')
      .attr('x', medianX - 6)
      .attr('y', 14)
      .attr('text-anchor', 'end')
      .attr('opacity', 0)
      .text(`p50: ${medianVal}ms`)
      .transition().delay(1300).duration(300)
      .attr('opacity', 1);

    // Dim left half (after sort)
    chartG.selectAll('.bar')
      .filter((d) => {
        const sortIdx = sortedData.findIndex(s => s.idx === d.idx);
        return sortIdx < medianIdx;
      })
      .transition().delay(1200).duration(400)
      .style('opacity', 0.3);
  }

  /* ── Step 6: Show p99 ── */
  function showP99() {
    const sortedData = data._sorted;
    // With only 30 data points, position p99 at the last normal bar (before outliers)
    const p99Idx = sortedData.findIndex(d => d.value >= OUTLIER_THRESHOLD) - 1;
    const p99Val = sortedData[p99Idx].value;
    const p99X = xScale(sortedData[p99Idx].idx) + xScale.bandwidth() / 2;

    // Undim bars between p50 and p99 (restore opacity)
    const medianIdx = Math.floor(sortedData.length / 2);
    chartG.selectAll('.bar')
      .filter((d) => {
        const sortIdx = sortedData.findIndex(s => s.idx === d.idx);
        return sortIdx >= medianIdx && sortIdx <= p99Idx;
      })
      .transition().duration(300)
      .style('opacity', 0.7);

    // Vertical p99 line
    chartG.append('line')
      .attr('class', 'p99-line')
      .attr('x1', p99X).attr('x2', p99X)
      .attr('y1', height)
      .attr('y2', height)
      .transition().duration(500)
      .attr('y2', 0);

    // p99 label — offset below p50 label to avoid overlap
    chartG.append('text')
      .attr('class', 'p99-label')
      .attr('x', p99X - 6)
      .attr('y', 30)
      .attr('text-anchor', 'end')
      .attr('opacity', 0)
      .text(`p99: ${p99Val}ms`)
      .transition().delay(400).duration(300)
      .attr('opacity', 1);

    // Color outlier bars red
    chartG.selectAll('.bar')
      .filter((d) => {
        const sortIdx = sortedData.findIndex(s => s.idx === d.idx);
        return sortIdx > p99Idx;
      })
      .classed('bar-worst', true)
      .transition().duration(400)
      .style('opacity', 1);

    // "Worst 1%" annotation
    const outlierBars = sortedData.filter(d => d.value >= OUTLIER_THRESHOLD);
    const midOutlierX = xScale(outlierBars[Math.floor(outlierBars.length / 2)].idx) + xScale.bandwidth() / 2;
    chartG.append('text')
      .attr('class', 'annotation-text')
      .attr('x', midOutlierX)
      .attr('y', yScale(outlierBars[0].value) - 12)
      .attr('text-anchor', 'middle')
      .attr('opacity', 0)
      .text('Worst 1%')
      .style('fill', '#ffd700')
      .style('font-weight', '700')
      .style('font-size', '12px')
      .transition().delay(600).duration(300)
      .attr('opacity', 1);
  }

  /* ── Step 7: Show SLO line ── */
  function showSLO() {
    const sloVal = 200;
    const sloY = yScale(sloVal);

    // SLO horizontal line
    chartG.append('line')
      .attr('class', 'slo-line')
      .attr('x1', 0).attr('y1', sloY)
      .attr('x2', 0).attr('y2', sloY)
      .transition().duration(600)
      .attr('x2', width);

    // SLO label — positioned above the line
    chartG.append('text')
      .attr('class', 'slo-label')
      .attr('x', width - 4)
      .attr('y', sloY - 8)
      .attr('text-anchor', 'end')
      .text('SLO: p99 < 200ms')
      .transition().delay(500).duration(300)
      .attr('opacity', 1);

    // Checkmark
    const sortedData = data._sorted;
    const p99Idx = sortedData.findIndex(d => d.value >= OUTLIER_THRESHOLD) - 1;
    const p99Val = sortedData[p99Idx].value;
    if (p99Val <= sloVal) {
      const p99X = xScale(sortedData[p99Idx].idx) + xScale.bandwidth() / 2;
      chartG.append('text')
        .attr('class', 'slo-check')
        .attr('x', p99X + 16)
        .attr('y', yScale(p99Val) + 8)
        .attr('opacity', 0)
        .text('✓')
        .transition().delay(800).duration(300)
        .attr('opacity', 1);
    }
  }

  /* ── Scene management ── */
  function showScene(el) {
    allScenes.forEach(s => s.classList.remove('active'));
    el.classList.add('active');
  }

  function updateCounter() {
    stepCounter.textContent = `${currentStep} / ${TOTAL_STEPS}`;
  }

  /* ── Step handlers ── */
  function step1() {
    showScene(sceneHook);
  }

  function step2() {
    showScene(sceneChart);
    chartTitle.classList.add('visible');
    if (!svg) initChart();
    drawBars();
  }

  function step3() { showAverage(); }
  function step4() { sortBarsAndShowP50(); }
  function step5() { showP99(); }
  function step6() { showSLO(); }

  function step7() {
    showScene(sceneCompare);
  }

  function step8() {
    showScene(sceneTakeaway);
  }

  const steps = [null, step1, step2, step3, step4, step5, step6, step7, step8];

  function advance() {
    if (currentStep >= TOTAL_STEPS) return;
    currentStep++;
    steps[currentStep]();
    updateCounter();
    if (currentStep >= TOTAL_STEPS) {
      clickHint.textContent = '← Tap to restart';
    }
  }

  function reset() {
    currentStep = 0;
    sorted = false;
    data = shuffledData.map((v, i) => ({ value: v, idx: i }));
    delete data._sorted;

    allScenes.forEach(s => s.classList.remove('active'));
    chartTitle.classList.remove('visible');

    // Clear chart
    if (svg) {
      svg.remove();
      svg = null;
      chartG = null;
    }

    clickHint.textContent = 'Tap or press → to advance';
    updateCounter();
  }

  /* ── Debounce ── */
  let lastAdvance = 0;
  const DEBOUNCE_MS = 350;

  function handleAdvance() {
    const now = Date.now();
    if (now - lastAdvance < DEBOUNCE_MS) return;
    lastAdvance = now;

    if (currentStep >= TOTAL_STEPS) {
      reset();
    } else {
      advance();
    }
  }

  stage.addEventListener('click', handleAdvance);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === ' ') {
      e.preventDefault();
      handleAdvance();
    }
    if (e.key === 'ArrowLeft' || e.key === 'r') {
      reset();
    }
  });

  updateCounter();
})();
