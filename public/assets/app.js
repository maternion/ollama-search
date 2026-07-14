// ollama-search frontend logic.
// Filtering, sorting, dark-mode, tab switching, copy-to-clipboard.

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
  var input = document.getElementById('form-input') || document.getElementById('navbar-input');
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
    'tags': 'data-sizes-count',
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

document.addEventListener('DOMContentLoaded', function() {
  var desktopSort = document.getElementById('desktop-sort-select');
  var mobileSort = document.getElementById('mobile-sort-select');
  if (desktopSort && mobileSort) {
    syncSort(desktopSort, mobileSort);
    syncSort(mobileSort, desktopSort);
  }

  if (document.getElementById('card-list')) {
    var formInput = document.getElementById('form-input');
    var navInput = document.getElementById('navbar-input');
    if (formInput) formInput.addEventListener('input', applyFilters);
    if (navInput) navInput.addEventListener('input', applyFilters);
    document.querySelectorAll('.cap-filter').forEach(function(cb) { cb.addEventListener('change', applyFilters); });
    var cloudFilter = document.getElementById('cloud-filter');
    if (cloudFilter) cloudFilter.addEventListener('change', applyFilters);
    if (desktopSort) desktopSort.addEventListener('change', applyFilters);
    if (mobileSort) mobileSort.addEventListener('change', applyFilters);
    // read ?q= from URL query string
    var params = new URLSearchParams(location.search);
    var q = params.get('q');
    if (q) {
      if (formInput) formInput.value = q;
      if (navInput) navInput.value = q;
    }
    applyFilters();
  }
  initFmtFilters();
});
