// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const uploadProgress = document.getElementById('upload-progress-container');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const documentList = document.getElementById('document-list');
const docCount = document.getElementById('doc-count');
const clearAllBtn = document.getElementById('clear-all-btn');
const queryForm = document.getElementById('query-form');
const queryInput = document.getElementById('query-input');
const chatContainer = document.getElementById('chat-container');
const clearChatBtn = document.getElementById('clear-chat-btn');
const topKSlider = document.getElementById('top-k-slider');
const topKVal = document.getElementById('top-k-val');
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdVal = document.getElementById('threshold-val');
const statLlm = document.getElementById('stat-llm');
const statEmb = document.getElementById('stat-emb');
const statChunks = document.getElementById('stat-chunks');
const toastEl = document.getElementById('toast');

// Toast notifications
function showToast(message, isError = false) {
    toastEl.textContent = message;
    toastEl.style.borderColor = isError ? 'var(--danger)' : 'var(--accent)';
    toastEl.classList.add('show');
    setTimeout(() => {
        toastEl.classList.remove('show');
    }, 3500);
}

// Load stats and documents on startup
async function refreshDashboard() {
    try {
        const [docsRes, statsRes] = await Promise.all([
            fetch('/api/v1/documents'),
            fetch('/api/v1/stats')
        ]);
        
        if (docsRes.ok) {
            const docs = await docsRes.json();
            renderDocuments(docs);
        }

        if (statsRes.ok) {
            const stats = await statsRes.json();
            statLlm.textContent = stats.active_model || stats.llm_provider;
            statEmb.textContent = stats.embeddings_provider;
            statChunks.textContent = stats.total_chunks;
        }
    } catch (e) {
        console.error('Failed to refresh dashboard:', e);
    }
}

function renderDocuments(docs) {
    docCount.textContent = docs.length;
    if (docs.length === 0) {
        documentList.innerHTML = '<div class="empty-state">No documents indexed yet. Upload one above!</div>';
        return;
    }

    documentList.innerHTML = docs.map(doc => `
        <div class="doc-card" id="doc-${doc.id}">
            <div class="doc-info">
                <span class="doc-name" title="${doc.filename}">${doc.filename}</span>
                <span class="doc-meta">${doc.chunk_count} chunks • ${(doc.file_size_bytes / 1024).toFixed(1)} KB</span>
            </div>
            <button class="btn-delete-doc" onclick="deleteDocument('${doc.id}', '${doc.filename}')" title="Delete document">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="3 6 5 6 21 6"></polyline>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                </svg>
            </button>
        </div>
    `).join('');
}

// File Upload Handlers
browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

dropZone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
});

['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });
});

dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        uploadFile(fileInput.files[0]);
    }
});

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    uploadProgress.style.display = 'block';
    progressBar.style.width = '30%';
    progressText.textContent = `Uploading ${file.name}...`;

    try {
        progressBar.style.width = '70%';
        progressText.textContent = `Extracting & embedding chunks...`;

        const res = await fetch('/api/v1/documents/upload', {
            method: 'POST',
            body: formData
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Upload failed');

        progressBar.style.width = '100%';
        progressText.textContent = 'Completed!';
        showToast(`Indexed: ${file.name} (${data.document.chunk_count} chunks)`);
        await refreshDashboard();
    } catch (err) {
        showToast(err.message, true);
    } finally {
        setTimeout(() => {
            uploadProgress.style.display = 'none';
            progressBar.style.width = '0%';
            fileInput.value = '';
        }, 1500);
    }
}

// Document Management Actions
async function deleteDocument(docId, filename) {
    if (!confirm(`Delete "${filename}" and its indexed chunks?`)) return;

    try {
        const res = await fetch(`/api/v1/documents/${docId}`, { method: 'DELETE' });
        if (res.ok) {
            showToast(`Deleted ${filename}`);
            await refreshDashboard();
        } else {
            showToast('Failed to delete document', true);
        }
    } catch (e) {
        showToast('Error deleting document', true);
    }
}

clearAllBtn.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to clear ALL indexed documents and vector stores?')) return;
    try {
        const res = await fetch('/api/v1/clear', { method: 'DELETE' });
        if (res.ok) {
            showToast('Cleared all indexed documents');
            await refreshDashboard();
        }
    } catch (e) {
        showToast('Error clearing documents', true);
    }
});

// Controls
topKSlider.addEventListener('input', (e) => {
    topKVal.textContent = e.target.value;
});

thresholdSlider.addEventListener('input', (e) => {
    thresholdVal.textContent = Number(e.target.value).toFixed(2);
});

// Chat & Query Handlers
queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    appendUserMessage(query);
    queryInput.value = '';
    queryInput.style.height = 'auto';

    const loadingId = appendLoadingMessage();

    try {
        const payload = {
            question: query,
            top_k: parseInt(topKSlider.value, 10),
            similarity_threshold: parseFloat(thresholdSlider.value)
        };

        const res = await fetch('/api/v1/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        removeMessage(loadingId);

        if (!res.ok) {
            appendAssistantMessage(`Error: ${data.detail || 'Failed to query system.'}`, null, false, 0);
            return;
        }

        appendAssistantMessage(
            data.answer,
            data.sources,
            data.has_grounded_context,
            data.confidence_score,
            data.latency_ms,
            data.model_used
        );
    } catch (err) {
        removeMessage(loadingId);
        appendAssistantMessage(`Connection error: ${err.message}`, null, false, 0);
    }
});

// Auto-expand textarea on typing & handle Enter key
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        queryForm.dispatchEvent(new Event('submit'));
    }
});

queryInput.addEventListener('input', () => {
    queryInput.style.height = 'auto';
    queryInput.style.height = queryInput.scrollHeight + 'px';
});

// Quick Prompts Click
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('prompt-chip')) {
        const prompt = e.target.getAttribute('data-prompt');
        queryInput.value = prompt;
        queryForm.dispatchEvent(new Event('submit'));
    }
});

clearChatBtn.addEventListener('click', () => {
    chatContainer.innerHTML = `
        <div class="message assistant-msg">
            <div class="msg-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"></path>
                    <path d="M12 16v-4"></path>
                    <path d="M12 8h.01"></path>
                </svg>
            </div>
            <div class="msg-body">
                <div class="msg-author">RAG Assistant</div>
                <div class="msg-content">Chat cleared. Ready for your questions!</div>
            </div>
        </div>
    `;
});

function appendUserMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user-msg';
    msgDiv.innerHTML = `
        <div class="msg-avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
            </svg>
        </div>
        <div class="msg-body">
            <div class="msg-author">You</div>
            <div class="msg-content">${escapeHtml(text)}</div>
        </div>
    `;
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function appendLoadingMessage() {
    const id = 'loading-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant-msg';
    msgDiv.id = id;
    msgDiv.innerHTML = `
        <div class="msg-avatar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
            </svg>
        </div>
        <div class="msg-body">
            <div class="msg-author">RAG Assistant</div>
            <div class="msg-content" style="color: var(--text-muted); font-style: italic;">
                Retrieving chunks and synthesizing response...
            </div>
        </div>
    `;
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function appendAssistantMessage(answer, sources, isGrounded, confidence, latencyMs = 0, model = '') {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant-msg';

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="sources-container">
                <div class="sources-header" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'flex' : 'none'">
                    <span>Source Citations (${sources.length} chunks referenced)</span>
                    <span style="font-size: 0.7rem;">▼ Click to toggle</span>
                </div>
                <div class="sources-list">
                    ${sources.map((s, idx) => `
                        <div class="source-item">
                            <div class="source-item-meta">
                                [${idx + 1}] ${escapeHtml(s.document_name)} • Page ${s.page || 1} • Similarity: ${(s.similarity_score * 100).toFixed(1)}%
                            </div>
                            <div class="source-item-snippet">"${escapeHtml(s.content.substring(0, 240))}..."</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    const groundPill = isGrounded 
        ? `<span class="metric-pill grounded">✓ Grounded in Document</span>`
        : `<span class="metric-pill ungrounded">⚠ Low/No Context</span>`;

    const confPercent = (confidence * 100).toFixed(0);

    msgDiv.innerHTML = `
        <div class="msg-avatar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"></path>
                <path d="M12 16v-4"></path>
                <path d="M12 8h.01"></path>
            </svg>
        </div>
        <div class="msg-body">
            <div class="msg-author">RAG Assistant</div>
            <div class="msg-content">${escapeHtml(answer)}</div>
            <div class="msg-metrics">
                ${groundPill}
                <span class="metric-pill">Confidence: ${confPercent}%</span>
                <span class="metric-pill">${latencyMs}ms</span>
                ${model ? `<span class="metric-pill">${model}</span>` : ''}
            </div>
            ${sourcesHtml}
        </div>
    `;

    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initial dashboard load
refreshDashboard();
