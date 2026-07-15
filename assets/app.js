// ollama-search frontend logic.
// Filtering, sorting, dark-mode, tab switching, copy-to-clipboard.

var NAV_MODELS = null;
var NAV_BASE = (function() {
  var s = document.querySelector('script[src*="assets/app.js"]');
  if (s) {
    var src = s.getAttribute('src');
    var idx = src.indexOf('assets/app.js');
    if (idx > 0) return src.substring(0, idx);
  }
  return '/';
})();

function loadNavModels(cb) {
  if (NAV_MODELS) { cb(NAV_MODELS); return; }
  var cached = null;
  try { cached = sessionStorage.getItem('nav-models'); } catch (e) {}
  if (cached) {
    try { NAV_MODELS = JSON.parse(cached); cb(NAV_MODELS); return; } catch (e) {}
  }
  fetch(NAV_BASE + 'assets/models.json').then(function(r) { return r.json(); }).then(function(data) {
    NAV_MODELS = data;
    try { sessionStorage.setItem('nav-models', JSON.stringify(data)); } catch (e) {}
    cb(NAV_MODELS);
  }).catch(function() { cb([]); });
}

function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function renderNavSuggest(query) {
  var sp = document.getElementById('searchpreview');
  if (!sp) return;
  var q = query.toLowerCase().trim();
  if (!q) { sp.classList.add('hidden'); sp.innerHTML = ''; return; }

  var html = '<div class="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl w-full shadow-2xl shadow-black/5 overflow-hidden" id="search-preview-container" tabindex="0">';
  html += '<div role="list" id="search-preview-list" class="group">';

  var results = [];
  for (var i = 0; i < NAV_MODELS.length; i++) {
    var m = NAV_MODELS[i];
    var name = m.name || '';
    var desc = m.description || '';
    if (name.toLowerCase().indexOf(q) !== -1 || desc.toLowerCase().indexOf(q) !== -1) {
      results.push(m);
    }
  }
  results.sort(function(a, b) { return (b.pulls || 0) - (a.pulls || 0); });
  var top = results.slice(0, 5);

  if (top.length === 0) {
    html += '<div class="px-6 py-4 text-neutral-800 dark:text-neutral-300 text-sm">No models found.</div>';
  } else {
    for (var i = 0; i < top.length; i++) {
      var m = top[i];
      var path = m.path || ('/library/' + m.name);
      html += '<div result>';
      html += '<a tabindex="0" href="' + NAV_BASE + path.replace(/^\//, '') + '" class="flex items-center h-16 px-6 py-4 hover:bg-neutral-50 dark:hover:bg-white/5 focus:ring-0 focus:outline-none focus:bg-neutral-50 dark:focus:bg-white/5">';
      html += '<div class="min-w-0 flex-1">';
      html += '<h2 class="text-sm font-medium truncate dark:text-neutral-100">' + escHtml(m.name) + '</h2>';
      html += '<p class="text-xs text-gray-600 dark:text-gray-600 truncate">' + escHtml(m.description) + '</p>';
      html += '</div></a></div>';
    }
  }

  html += '</div>';
  html += '<a tabindex="0" id="view-all-link" href="' + NAV_BASE + '?q=' + encodeURIComponent(query) + '" class="' + (top.length === 0 ? 'hidden' : '') + ' block px-6 py-3 border-t border-neutral-200 dark:border-neutral-800 text-center text-sm font-semibold hover:bg-neutral-50 dark:hover:bg-white/5 focus:bg-neutral-50 dark:focus:bg-white/5 focus:outline-none focus:ring-0 dark:text-neutral-200">View all &#8594;</a>';
  html += '</div>';

  sp.innerHTML = html;
  sp.classList.remove('hidden');
}

var navSuggestTimer = null;
function initNavSuggest() {
  var input = document.getElementById('navbar-input');
  var sp = document.getElementById('searchpreview');
  if (!input || !sp) return;

  input.addEventListener('input', function() {
    if (navSuggestTimer) clearTimeout(navSuggestTimer);
    navSuggestTimer = setTimeout(function() {
      var v = input.value;
      if (!v.trim()) { sp.classList.add('hidden'); sp.innerHTML = ''; return; }
      loadNavModels(function() { renderNavSuggest(v); });
    }, 100);
  });

  sp.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      sp.classList.add('hidden');
      input.focus();
      e.preventDefault();
      return;
    }
    if (e.key === 'Enter') {
      var el = document.activeElement;
      if (el && el.tagName === 'A') { el.click(); e.preventDefault(); return; }
    }
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      var items = Array.from(sp.querySelectorAll('#search-preview-list a, #view-all-link'));
      var ci = items.indexOf(document.activeElement);
      var ni = e.key === 'ArrowDown' ? ci + 1 : ci - 1;
      if (ni >= items.length) ni = 0;
      if (ni < 0) ni = items.length - 1;
      if (items[ni]) items[ni].focus();
      e.preventDefault();
    }
  });
}

function copyToClipboard(btn) {
  var input = btn.parentElement.querySelector('input.command');
  if (!input) return;
  navigator.clipboard.writeText(input.value).then(function() {
    var copyIcon = btn.querySelector('.copy-icon');
    var checkIcon = btn.querySelector('.check-icon');
    if (copyIcon) copyIcon.classList.add('hidden');
    if (checkIcon) checkIcon.classList.remove('hidden');
    setTimeout(function() {
      if (copyIcon) copyIcon.classList.remove('hidden');
      if (checkIcon) checkIcon.classList.add('hidden');
    }, 1500);
  });
}
window.copyToClipboard = copyToClipboard;

// --- Search page: filter + sort + capability chips ---

function getSelectedCaps() {
  var caps = [];
  document.querySelectorAll('.cap-filter').forEach(function(cb) {
    if (cb.checked) caps.push(cb.getAttribute('data-cap'));
  });
  return caps;
}

function getSort() {
  var sortEl = document.getElementById('desktop-sort-select') || document.getElementById('mobile-sort-select');
  return sortEl ? sortEl.value : 'popular';
}

function getQuery() {
  var a = document.activeElement;
  var input = (a && (a.id === 'form-input' || a.id === 'navbar-input'))
    ? a
    : (document.getElementById('form-input') || document.getElementById('navbar-input'));
  return input ? input.value.toLowerCase().trim() : '';
}

function getCloudFilter() {
  var el = document.getElementById('cloud-filter');
  return el ? el.value : 'all';
}

function applyFilters() {
  var q = getQuery();
  var caps = getSelectedCaps();
  var sort = getSort();
  var cloudFilter = getCloudFilter();
  var list = document.getElementById('card-list');
  if (!list) return;
  var cards = Array.from(list.querySelectorAll('li[x-test-model]'));
  // Filter
  var visible = 0;
  cards.forEach(function(card) {
    var title = card.querySelector('[x-test-search-response-title]') ? card.querySelector('[x-test-search-response-title]').textContent.toLowerCase() : '';
    var desc = card.querySelector('p.break-words') ? card.querySelector('p.break-words').textContent.toLowerCase() : '';
    var cardCaps = [];
    card.querySelectorAll('[x-test-capability]').forEach(function(el) { cardCaps.push(el.textContent.toLowerCase()); });
    var isCloud = card.getAttribute('data-cloud') === 'true';
    var isCloudOnly = card.getAttribute('data-cloud-only') === 'true';
    var matchText = !q || title.indexOf(q) !== -1 || desc.indexOf(q) !== -1;
    var matchCaps = caps.length === 0 || caps.every(function(c) { return cardCaps.indexOf(c) !== -1; });
    var matchCloud = cloudFilter === 'all'
      || (cloudFilter === 'cloud' && isCloud)
      || (cloudFilter === 'local' && !isCloudOnly);
    var show = matchText && matchCaps && matchCloud;
    card.style.display = show ? '' : 'none';
    if (show) visible++;
  });
  var noRes = document.getElementById('no-results');
  if (noRes) noRes.classList.toggle('hidden', visible > 0);
  // Sort — always reorder DOM using data-* rank attributes
  var rankAttr = {
    'popular': 'data-popular-rank',
    'newest': 'data-newest-rank',
    'oldest': 'data-oldest-rank',
    'updated': 'data-updated-rank',
    'pulls': 'data-pulls',
    'tags': 'data-tag-count',
    'name': 'data-name',
  };
  var attr = rankAttr[sort] || rankAttr['popular'];
  var descending = (sort === 'pulls' || sort === 'tags');
  cards.sort(function(a, b) {
    var va = a.getAttribute(attr) || '';
    var vb = b.getAttribute(attr) || '';
    var cmp;
    if (sort === 'name') {
      cmp = va.localeCompare(vb);
    } else {
      var na = parseFloat(va) || 0;
      var nb = parseFloat(vb) || 0;
      cmp = na - nb;
      if (descending) cmp = -cmp;
    }
    return cmp;
  });
  cards.forEach(function(c) { list.appendChild(c); });
}

// --- Usage section: tab switching + copy ---
function switchUsageTab(btn, tabName) {
  var section = btn.closest('section');
  section.querySelectorAll('.use-tab').forEach(function(tab) {
    tab.classList.remove('text-neutral-900', 'font-medium', 'underline', 'decoration-1', 'underline-offset-[7px]');
    tab.classList.add('text-neutral-400');
  });
  btn.classList.remove('text-neutral-400');
  btn.classList.add('text-neutral-900', 'font-medium', 'underline', 'decoration-1', 'underline-offset-[7px]');
  section.querySelectorAll('.use-panel').forEach(function(panel) { panel.classList.add('hidden'); });
  var activePanel = section.querySelector('.use-panel[data-panel="' + tabName + '"]');
  if (activePanel) activePanel.classList.remove('hidden');
  section.querySelectorAll('.use-link').forEach(function(link) { link.classList.add('hidden'); });
  var activeLink = section.querySelector('.use-link[data-link="' + tabName + '"]');
  if (activeLink) activeLink.classList.remove('hidden');
}
window.switchUsageTab = switchUsageTab;

function copyUsageCode(btn) {
  var section = btn.closest('section');
  var activePanel = section.querySelector('.use-panel:not(.hidden)');
  if (!activePanel) return;
  var pre = activePanel.querySelector('pre');
  if (!pre) return;
  navigator.clipboard.writeText(pre.textContent).then(function() {
    var copyIcon = btn.querySelector('.copy-icon');
    var checkIcon = btn.querySelector('.check-icon');
    if (copyIcon) copyIcon.classList.add('hidden');
    if (checkIcon) checkIcon.classList.remove('hidden');
    setTimeout(function() {
      if (copyIcon) copyIcon.classList.remove('hidden');
      if (checkIcon) checkIcon.classList.add('hidden');
    }, 2000);
  });
}
window.copyUsageCode = copyUsageCode;

// --- Format pill radio filters (detail + tags pages) ---
function initFmtFilters() {
  var radios = document.querySelectorAll('.fmt-radio');
  if (!radios.length) return;
  radios.forEach(function(radio) {
    radio.addEventListener('change', function() {
      var fmt = radio.getAttribute('data-fmt');
      document.querySelectorAll('.fmt-table').forEach(function(tbl) {
        var id = tbl.id.replace('tags-table-', '').replace('models-table-', '');
        tbl.classList.toggle('hidden', id !== fmt);
      });
    });
  });
}

// Sync mobile and desktop sort selects
function syncSort(source, target) {
  if (source && target) {
    source.addEventListener('change', function() { target.value = source.value; applyFilters(); });
  }
}

function initApp() {
  var desktopSort = document.getElementById('desktop-sort-select');
  var mobileSort = document.getElementById('mobile-sort-select');
  if (desktopSort && mobileSort) {
    syncSort(desktopSort, mobileSort);
    syncSort(mobileSort, desktopSort);
  }

  if (document.getElementById('card-list')) {
    var formInput = document.getElementById('form-input');
    var navInput = document.getElementById('navbar-input');
    if (formInput) formInput.addEventListener('input', function() {
      if (navInput) navInput.value = formInput.value;
      applyFilters();
    });
    if (navInput) navInput.addEventListener('input', function() {
      if (formInput) formInput.value = navInput.value;
      applyFilters();
    });
    document.querySelectorAll('.cap-filter').forEach(function(cb) { cb.addEventListener('change', applyFilters); });
    var cloudFilter = document.getElementById('cloud-filter');
    if (cloudFilter) cloudFilter.addEventListener('change', applyFilters);
    // read ?q= from URL query string
    var params = new URLSearchParams(location.search);
    var q = params.get('q');
    if (q) {
      if (formInput) formInput.value = q;
      if (navInput) navInput.value = q;
    }
    applyFilters();
  }

  // --- Navbar search preview dropdown (non-search pages only) ---
  if (!document.getElementById('card-list')) {
    initNavSuggest();
  }
  initFmtFilters();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
