/* LEO conversation module — markdown renderer for assistant output.
   Ported from the Hermes dashboard's Markdown.tsx block parser (code fences,
   headings, hr, lists, paragraphs) plus blockquotes, tables and inline
   formatting. Builds DOM nodes only — model output is never trusted as HTML.
   Exposed as window.LEOMarkdown. */

(() => {
  'use strict';

  /* ---- inline formatting -------------------------------------------------
     Walks a plain-text string and returns DOM nodes. Code spans are extracted
     first (their contents must never be formatted); then bold/italic/links. */

  const CODE_SPAN = /(`+)([\s\S]*?)\1(?!`)/;
  const LINK = /\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/;
  const AUTOLINK = /https?:\/\/[^\s<>"')\]]+/;
  const BOLD = /\*\*([^*]+)\*\*|__([^_]+)__/;
  const ITALIC = /\*([^*\s][^*]*?)\*|_([^_\s][^_]*?)_/;

  function safeHref(url) {
    const trimmed = String(url || '').trim();
    if (/^(https?:|mailto:)/i.test(trimmed)) return trimmed;
    if (trimmed.startsWith('/') && !trimmed.startsWith('//')) return trimmed;
    if (trimmed.startsWith('#')) return trimmed;
    return null;
  }

  function makeLink(href, textContent) {
    const a = document.createElement('a');
    a.href = href;
    a.textContent = textContent;
    a.target = '_blank';
    a.rel = 'noopener noreferrer nofollow';
    return a;
  }

  function inlineNodes(text, target) {
    let rest = String(text ?? '');
    while (rest) {
      // code span
      let m = CODE_SPAN.exec(rest);
      if (m) {
        if (m.index > 0) inlineFormatted(rest.slice(0, m.index), target);
        const code = document.createElement('code');
        code.className = 'inline';
        code.textContent = m[2];
        target.appendChild(code);
        rest = rest.slice(m.index + m[0].length);
        continue;
      }
      // markdown link
      m = LINK.exec(rest);
      if (m) {
        if (m.index > 0) inlineFormatted(rest.slice(0, m.index), target);
        const href = safeHref(m[2]);
        if (href) target.appendChild(makeLink(href, m[1] || m[2]));
        else target.appendChild(document.createTextNode(m[0]));
        rest = rest.slice(m.index + m[0].length);
        continue;
      }
      // bare URL
      m = AUTOLINK.exec(rest);
      if (m) {
        if (m.index > 0) inlineFormatted(rest.slice(0, m.index), target);
        target.appendChild(makeLink(m[0], m[0]));
        rest = rest.slice(m.index + m[0].length);
        continue;
      }
      inlineFormatted(rest, target);
      rest = '';
    }
  }

  function inlineFormatted(text, target) {
    let rest = text;
    while (rest) {
      const bold = BOLD.exec(rest);
      const ital = ITALIC.exec(rest);
      const pick = bold && (!ital || bold.index <= ital.index) ? bold : ital;
      if (!pick) { target.appendChild(document.createTextNode(rest)); return; }
      if (pick.index > 0) target.appendChild(document.createTextNode(rest.slice(0, pick.index)));
      const el = document.createElement(pick === bold ? 'strong' : 'em');
      el.textContent = pick[1] ?? pick[2] ?? '';
      target.appendChild(el);
      rest = rest.slice(pick.index + pick[0].length);
    }
  }

  /* ---- block parsing ------------------------------------------------------ */

  const FENCE = /^(```|~~~)\s*([\w+-]*)/;
  const HEADING = /^(#{1,4})\s+(.*)$/;
  const HR = /^\s*([-*_])(\s*\1){2,}\s*$/;
  const UL = /^\s*[-*+]\s+/;
  const OL = /^\s*\d+[.)]\s+/;
  const QUOTE = /^\s*>\s?/;
  const TABLE_DIVIDER = /^\s*\|?[\s:|-]+\|[\s:|-]*$/;

  function splitTableRow(line) {
    return line.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map((cell) => cell.trim());
  }

  function buildCodeBlock(lang, code) {
    const wrap = document.createElement('div');
    wrap.className = 'md-code';
    const head = document.createElement('div');
    head.className = 'md-code__head';
    const label = document.createElement('span');
    label.className = 'md-code__lang';
    label.textContent = lang || 'text';
    const copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'md-code__copy';
    copy.textContent = 'copy';
    copy.addEventListener('click', () => {
      navigator.clipboard.writeText(code).then(
        () => { copy.textContent = 'copied'; setTimeout(() => { copy.textContent = 'copy'; }, 1600); },
        () => { copy.textContent = 'failed'; setTimeout(() => { copy.textContent = 'copy'; }, 1600); },
      );
    });
    head.append(label, copy);
    const pre = document.createElement('pre');
    const codeEl = document.createElement('code');
    codeEl.textContent = code;
    pre.appendChild(codeEl);
    wrap.append(head, pre);
    return wrap;
  }

  function buildParagraph(lines) {
    const p = document.createElement('p');
    inlineNodes(lines.join('\n'), p);
    return p;
  }

  function buildList(lines, ordered) {
    const list = document.createElement(ordered ? 'ol' : 'ul');
    for (const line of lines) {
      const li = document.createElement('li');
      inlineNodes(line, li);
      list.appendChild(li);
    }
    return list;
  }

  function buildTable(headerLine, dividerLine, bodyLines) {
    const headers = splitTableRow(headerLine);
    const align = splitTableRow(dividerLine).map((cell) =>
      cell.startsWith(':') && cell.endsWith(':') ? 'center' : cell.endsWith(':') ? 'right' : 'left');
    const table = document.createElement('table');
    table.className = 'md-table';
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    headers.forEach((cell, i) => {
      const th = document.createElement('th');
      th.textContent = cell;
      if (align[i] && align[i] !== 'left') th.style.textAlign = align[i];
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    const tbody = document.createElement('tbody');
    for (const line of bodyLines) {
      const cells = splitTableRow(line);
      const row = document.createElement('tr');
      for (let i = 0; i < headers.length; i += 1) {
        const td = document.createElement('td');
        inlineNodes(cells[i] ?? '', td);
        if (align[i] && align[i] !== 'left') td.style.textAlign = align[i];
        row.appendChild(td);
      }
      tbody.appendChild(row);
    }
    table.append(thead, tbody);
    return table;
  }

  function blocks(text) {
    const out = [];
    const lines = String(text ?? '').split('\n');
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i += 1; continue; }

      const fence = FENCE.exec(line);
      if (fence) {
        const lang = fence[2] || '';
        const body = [];
        i += 1;
        while (i < lines.length && !FENCE.test(lines[i])) { body.push(lines[i]); i += 1; }
        i += 1;  // closing fence (or EOF)
        out.push({ kind: 'code', node: buildCodeBlock(lang, body.join('\n')) });
        continue;
      }

      const heading = HEADING.exec(line);
      if (heading) {
        const el = document.createElement(`h${Math.min(heading[1].length, 4)}`);
        inlineNodes(heading[2], el);
        out.push({ kind: 'inline-block', node: el });
        i += 1;
        continue;
      }

      if (HR.test(line)) { out.push({ kind: 'block', node: document.createElement('hr') }); i += 1; continue; }

      if (QUOTE.test(line)) {
        const body = [];
        while (i < lines.length && QUOTE.test(lines[i])) {
          body.push(lines[i].replace(QUOTE, ''));
          i += 1;
        }
        const q = document.createElement('blockquote');
        const inner = document.createElement('div');
        inner.className = 'md';
        inner.style.gap = '4px';
        for (const part of blocks(body.join('\n'))) inner.appendChild(part.node);
        q.appendChild(inner);
        out.push({ kind: 'block', node: q });
        continue;
      }

      // GFM table: header row + |---|---| divider + body rows
      if (
        line.includes('|') && i + 1 < lines.length &&
        TABLE_DIVIDER.test(lines[i + 1]) && lines[i + 1].includes('-')
      ) {
        const body = [];
        let j = i + 2;
        while (j < lines.length && lines[j].trim() && lines[j].includes('|')) { body.push(lines[j]); j += 1; }
        out.push({ kind: 'block', node: buildTable(line, lines[i + 1], body) });
        i = j;
        continue;
      }

      if (UL.test(line) || OL.test(line)) {
        const ordered = OL.test(line);
        const strip = ordered ? OL : UL;
        const items = [];
        while (i < lines.length && strip.test(lines[i]) && line.trim()) {
          items.push(lines[i].replace(strip, '').trim());
          i += 1;
        }
        out.push({ kind: 'block', node: buildList(items, ordered) });
        continue;
      }

      const para = [];
      while (
        i < lines.length && lines[i].trim() &&
        !FENCE.test(lines[i]) && !HEADING.test(lines[i]) && !HR.test(lines[i]) &&
        !QUOTE.test(lines[i]) && !UL.test(lines[i]) && !OL.test(lines[i])
      ) { para.push(lines[i]); i += 1; }
      if (para.length) out.push({ kind: 'inline-block', node: buildParagraph(para) });
      else i += 1;
    }
    return out;
  }

  /* ---- public API ---------------------------------------------------------- */

  function renderInto(container, text, { streaming = false } = {}) {
    const host = typeof container === 'string' ? document.getElementById(container) : container;
    if (!host) return;
    host.replaceChildren();
    const md = document.createElement('div');
    md.className = 'md';
    const parsed = blocks(text);
    parsed.forEach((entry, index) => md.appendChild(entry.node));
    if (streaming) {
      const caret = document.createElement('span');
      caret.className = 'md-caret';
      caret.setAttribute('aria-hidden', 'true');
      const last = parsed[parsed.length - 1];
      if (last && last.kind === 'inline-block') last.node.appendChild(caret);
      else if (last && last.kind === 'code') {
        // An open (still streaming) fence renders as a code block; place the
        // caret inside the code element so it sits at the actual typing point.
        const codeEl = last.node.querySelector('pre code');
        if (codeEl) codeEl.appendChild(caret);
        else md.appendChild(caret);
      } else md.appendChild(caret);
    }
    host.appendChild(md);
  }

  window.LEOMarkdown = { renderInto };
})();
