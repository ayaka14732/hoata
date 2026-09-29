import { useCallback, useEffect, useRef, useState } from 'react'
import type { CSSProperties, PointerEvent } from 'react'
import { ArrowDownToLine, ArrowRight, ChevronDown, ChevronLeft, ChevronRight, Search, X } from 'lucide-react'
import { CodeEditor } from './CodeEditor'
import { Select } from './Select'
import type { EditorHandle } from './CodeEditor'
import { Engine } from './engine'
import { initialLanguage, messages } from './i18n'
import { emptySelection } from './types'
import type { Compilation, Config, ExampleId, Language, Selection, Summary, Target } from './types'
import clamp from '../examples/clamp.py?raw'
import square from '../examples/square.py?raw'
import matmul from '../examples/matmul.py?raw'

const examples = { clamp, square, matmul }
const defaultDraft = { source: clamp, target: 'tpu-v4-tc' as Target, example: 'clamp' as ExampleId }
function readDraft(): typeof defaultDraft {
  try {
    const saved = JSON.parse(localStorage.getItem('hoata.draft') ?? 'null')
    if (saved && typeof saved.source === 'string' && saved.source.length <= 262144 && ['tpu-v4-tc', 'tpu-v6e-tc'].includes(saved.target)) {
      return { source: saved.source, target: saved.target, example: saved.example in examples ? saved.example : 'clamp' }
    }
  } catch { /* Start with the bundled example when storage is unavailable. */ }
  return defaultDraft
}
function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }))
  const anchor = document.createElement('a')
  anchor.href = url; anchor.download = name; anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function App() {
  const [draft] = useState(readDraft)
  const [source, setSource] = useState(draft.source)
  const [target, setTarget] = useState<Target>(draft.target)
  const [example, setExample] = useState<ExampleId>(draft.example)
  const [language, setLanguage] = useState<Language>(initialLanguage)
  const t = messages[language]
  const dir = language === 'he' ? 'rtl' : 'ltr'
  const [config, setConfig] = useState<Config | null>(null)
  const [connectionError, setConnectionError] = useState(false)
  const [retry, setRetry] = useState(0)
  const [compilation, setCompilation] = useState<Compilation | null>(null)
  const [compiledSource, setCompiledSource] = useState('')
  const [compiledTarget, setCompiledTarget] = useState<Target>(target)
  const [busy, setBusy] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [programIndex, setProgramIndex] = useState(0)
  const [summary, setSummary] = useState<Summary | null>(null)
  const [selection, setSelection] = useState<Selection>(emptySelection)
  const [engineError, setEngineError] = useState('')
  const [search, setSearch] = useState('')
  const [matches, setMatches] = useState<number[]>([])
  const [matchIndex, setMatchIndex] = useState(0)
  const [panel, setPanel] = useState<'diagnostics' | 'output' | null>(null)
  const [ratio, setRatio] = useState(47)
  const [pendingExample, setPendingExample] = useState<ExampleId | null>(null)
  const dialog = useRef<HTMLDialogElement>(null)
  const workspace = useRef<HTMLDivElement>(null)
  const sourceEditor = useRef<EditorHandle>(null)
  const assemblyEditor = useRef<EditorHandle>(null)
  const engine = useRef<Engine | null>(null)
  const activeRequest = useRef<string | null>(null)
  const compileController = useRef<AbortController | null>(null)
  const selectionSerial = useRef(0)
  const stale = compilation !== null && (compiledSource !== source || compiledTarget !== target)
  const program = compilation?.ok ? compilation.programs[programIndex] : undefined
  const linked = Boolean(program && summary && !stale && !busy)
  const live = useRef({ linked, source, target, config })
  live.current = { linked, source, target, config }

  useEffect(() => {
    engine.current = new Engine()
    return () => {
      if (activeRequest.current) void fetch(`/api/compile/${activeRequest.current}`, { method: 'DELETE', keepalive: true })
      compileController.current?.abort()
      engine.current?.dispose()
    }
  }, [])
  useEffect(() => {
    document.documentElement.lang = language
    document.documentElement.dir = dir
    document.title = 'Hoata'
    try { localStorage.setItem('hoata.language', language) } catch { /* Optional persistence. */ }
  }, [language, dir])
  useEffect(() => {
    const timer = setTimeout(() => {
      try { localStorage.setItem('hoata.draft', JSON.stringify({ source, target, example })) }
      catch { /* Optional persistence. */ }
    }, 300)
    return () => clearTimeout(timer)
  }, [source, target, example])
  useEffect(() => {
    const controller = new AbortController()
    setConnectionError(false)
    fetch('/api/config', { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error('Server unavailable')
      setConfig(await response.json() as Config)
    }).catch(() => { if (!controller.signal.aborted) setConnectionError(true) })
    return () => controller.abort()
  }, [retry])
  useEffect(() => {
    if (!busy) return
    const started = Date.now()
    setElapsed(0)
    const timer = setInterval(() => setElapsed((Date.now() - started) / 1000), 100)
    return () => clearInterval(timer)
  }, [busy])
  useEffect(() => {
    let current = true
    ++selectionSerial.current
    setSummary(null); setSelection(emptySelection); setEngineError('')
    setSearch(''); setMatches([])
    if (program) {
      engine.current!.call<Summary>('load', JSON.stringify(program)).then(result => {
        if (!current) return
        setSummary(result)
        assemblyEditor.current?.reveal(result.preferred_line)
      }).catch(error => { if (current) setEngineError(String(error)) })
    }
    return () => { current = false }
  }, [program])
  useEffect(() => {
    sourceEditor.current?.decorate(linked ? summary : null, linked ? selection.source_lines : [])
    assemblyEditor.current?.decorate(summary, linked ? selection.assembly_lines : [], matches)
  }, [summary, selection, linked, matches])
  useEffect(() => {
    sourceEditor.current?.markErrors(!stale ? compilation?.diagnostics ?? [] : [])
  }, [compilation, stale])
  useEffect(() => {
    ++selectionSerial.current
    if (!linked) setSelection(emptySelection)
  }, [linked])
  useEffect(() => {
    let current = true
    if (!summary) return
    const timer = setTimeout(() => {
      engine.current!.call<number[]>('search', search).then(found => {
        if (!current) return
        setMatches(found); setMatchIndex(0)
        if (found.length) assemblyEditor.current?.reveal(found[0])
      }).catch(error => { if (current) setEngineError(String(error)) })
    }, 120)
    return () => { current = false; clearTimeout(timer) }
  }, [search, summary])
  useEffect(() => {
    if (pendingExample) dialog.current?.showModal()
    else dialog.current?.close()
  }, [pendingExample])
  useEffect(() => {
    if (engineError || summary?.diagnostics.length) setPanel('diagnostics')
  }, [engineError, summary])

  const compile = useCallback(async () => {
    if (activeRequest.current || !live.current.config) return
    const { source: submittedSource, target: submittedTarget } = live.current
    const id = crypto.randomUUID()
    activeRequest.current = id
    const controller = new AbortController()
    compileController.current = controller
    setBusy(true); setEngineError('')
    try {
      const response = await fetch('/api/compile', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, source: submittedSource, target: submittedTarget }), signal: controller.signal,
      })
      const result = await response.json() as Compilation
      if (activeRequest.current !== id) return
      setCompilation(result); setCompiledSource(submittedSource); setCompiledTarget(submittedTarget)
      setProgramIndex(0); setPanel(result.ok && !result.diagnostics.length ? null : 'diagnostics')
    } catch {
      if (controller.signal.aborted) return
      setCompilation({ ok: false, error: 'network', programs: [], diagnostics: [] })
      setCompiledSource(submittedSource); setCompiledTarget(submittedTarget); setPanel('diagnostics')
    } finally {
      if (activeRequest.current === id) { activeRequest.current = null; setBusy(false) }
    }
  }, [])
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); void compile() }
    }
    document.addEventListener('keydown', listener)
    return () => document.removeEventListener('keydown', listener)
  }, [compile])

  async function select(kind: 'source' | 'assembly', line: number) {
    if (!live.current.linked) return
    const serial = ++selectionSerial.current
    try {
      const result = await engine.current!.call<Selection>(kind, line)
      if (serial !== selectionSerial.current || !live.current.linked) return
      setSelection(result)
      if (kind === 'source' && result.assembly_lines.length) assemblyEditor.current?.reveal(result.assembly_lines[0])
      if (kind === 'assembly' && result.source_lines.length) sourceEditor.current?.reveal(result.source_lines[0])
    } catch (error) { setEngineError(String(error)) }
  }
  async function cancel() {
    if (activeRequest.current) await fetch(`/api/compile/${activeRequest.current}`, { method: 'DELETE' }).catch(() => {})
  }
  function loadExample(id: ExampleId) {
    setExample(id); setSource(examples[id]); setPendingExample(null)
  }
  function chooseExample(id: ExampleId) {
    if (source !== examples[example] && source !== examples[id]) setPendingExample(id)
    else loadExample(id)
  }
  function resize(event: PointerEvent<HTMLDivElement>) {
    event.currentTarget.setPointerCapture(event.pointerId)
    const rectangle = workspace.current!.getBoundingClientRect()
    const update = (pointer: globalThis.PointerEvent) => setRatio(Math.min(68, Math.max(32, (pointer.clientX - rectangle.left) / rectangle.width * 100)))
    const finish = () => { window.removeEventListener('pointermove', update); window.removeEventListener('pointerup', finish); window.removeEventListener('pointercancel', finish) }
    window.addEventListener('pointermove', update)
    window.addEventListener('pointerup', finish)
    window.addEventListener('pointercancel', finish)
  }
  function moveMatch(step: number) {
    if (!matches.length) return
    const next = (matchIndex + step + matches.length) % matches.length
    setMatchIndex(next); assemblyEditor.current?.reveal(matches[next])
  }
  const errorTitle = compilation?.error === 'cancelled' ? t.cancelled : compilation?.error === 'timeout' ? t.timeout : compilation?.error === 'busy' ? t.busy : compilation?.error === 'network' ? t.serverError : t.failed
  const status = busy ? t.compiling : stale ? t.stale : compilation && !compilation.ok ? errorTitle : ''
  const hasDiagnostics = Boolean(compilation?.error || compilation?.diagnostics.length || summary?.diagnostics.length || engineError)
  const output = [compilation?.logs?.stdout, compilation?.logs?.stderr, compilation?.traceback].filter(Boolean).join('\n')

  return <div className="app-shell">
    <header className="site-header">
      <div className="brand" dir="ltr">hoata<span className="brand-dot">.</span></div>
      <div className="compile-controls">
        <Select className="example-selector" label={t.example} value={source === examples[example] ? example : ''} placeholder={t.custom} dir={dir} onChange={value => chooseExample(value as ExampleId)} options={(Object.keys(examples) as ExampleId[]).map(id => ({ value: id, label: t[id] }))} />
        <Select className="target-control" testId="target" label={t.target} value={target} dir={dir} onChange={value => setTarget(value as Target)} options={[{ value: 'tpu-v4-tc', label: 'TPU v4' }, { value: 'tpu-v6e-tc', label: 'TPU v6e' }]} />
        {busy ? <button className="compile-button is-running" onClick={() => void cancel()}>{t.cancel}<span dir="ltr">{elapsed.toFixed(1)}s</span></button>
          : <button className="compile-button" data-testid="compile" onClick={() => void compile()} disabled={!config || !source.trim()} title={t.compileHint}>{t.compile}</button>}
      </div>
      <Select className="language-picker" label={t.language} value={language} dir={dir} onChange={value => setLanguage(value as Language)} options={[{ value: 'en', label: 'English' }, { value: 'zh', label: '中文' }, { value: 'he', label: 'עברית' }]} />
    </header>

    <main className="main-content">
      {connectionError && <div className="notice error"><span>{t.serverError}</span><button onClick={() => setRetry(value => value + 1)}>{t.retry}</button></div>}

      <div className="editor-workspace" ref={workspace} dir="ltr" style={{ '--source-width': `${ratio}%` } as CSSProperties}>
        <section className="editor-panel source-panel" dir={dir}>
          <div className="editor-panel-header"><bdi>kernel.py</bdi><button className="icon-button" onClick={() => download('kernel.py', source)} title={t.downloadSource} aria-label={t.downloadSource}><ArrowDownToLine size={15} /></button></div>
          <CodeEditor ref={sourceEditor} kind="source" value={source} label={t.sourceLabel} onChange={setSource} onSelect={line => void select('source', line)} onCompile={() => void compile()} />
        </section>

        <div className="resizer" role="separator" tabIndex={0} aria-label={t.resize} aria-orientation="vertical" aria-valuemin={32} aria-valuemax={68} aria-valuenow={Math.round(ratio)} onPointerDown={resize} onDoubleClick={() => setRatio(47)} onKeyDown={event => {
          if (event.key === 'ArrowLeft') { event.preventDefault(); setRatio(value => Math.max(32, value - 2)) }
          if (event.key === 'ArrowRight') { event.preventDefault(); setRatio(value => Math.min(68, value + 2)) }
          if (event.key === 'Home') { event.preventDefault(); setRatio(32) }
          if (event.key === 'End') { event.preventDefault(); setRatio(68) }
        }}><span /></div>

        <section className="editor-panel assembly-panel" dir={dir}>
          <div className="editor-panel-header"><span>{t.assembly}</span><span className={`result-status ${stale ? 'stale' : compilation && !compilation.ok ? 'failed' : ''}`} aria-live="polite" data-testid="compile-status">{status}</span><button className="icon-button" disabled={!program} onClick={() => program && download(`kernel-${program.target}-${program.id.replace(':', '-')}.tpuasm`, program.assembly)} title={t.downloadAssembly} aria-label={t.downloadAssembly}><ArrowDownToLine size={15} /></button></div>
          {program && <div className="assembly-tools">
            {compilation!.programs.length > 1 && <Select className="assembly-select" label={t.program} value={String(programIndex)} dir={dir} onChange={value => setProgramIndex(Number(value))} options={compilation!.programs.map((item, index) => ({ value: String(index), label: `${t.program} ${item.id}` }))} />}
            <Select className="assembly-select" label={t.function} value="" placeholder={t.function} dir={dir} disabled={!summary?.functions.some(fn => fn.lines.length)} onChange={value => {
              const fn = summary?.functions.find(item => String(item.id) === value)
              if (fn?.lines.length) assemblyEditor.current?.reveal(fn.lines[0])
            }} options={summary?.functions.filter(fn => fn.lines.length).map(fn => ({ value: String(fn.id), label: fn.name })) ?? []} />
            <div className="assembly-search"><Search size={13} /><input value={search} onChange={event => setSearch(event.target.value)} placeholder={t.search} aria-label={t.search} />{search && <><span className="search-count" dir="ltr">{matches.length ? `${matchIndex + 1}/${matches.length}` : '0'}</span><button onClick={() => moveMatch(-1)} disabled={!matches.length} aria-label={t.previousMatch}><ChevronLeft size={13} /></button><button onClick={() => moveMatch(1)} disabled={!matches.length} aria-label={t.nextMatch}><ChevronRight size={13} /></button><button onClick={() => setSearch('')} aria-label={t.dismiss}><X size={12} /></button></>}</div>
          </div>}
          <div className="assembly-body">
            <CodeEditor ref={assemblyEditor} kind="assembly" value={program?.assembly ?? ''} label={t.assemblyLabel} onSelect={line => void select('assembly', line)} onCompile={() => void compile()} />
            {!program && <div className="assembly-empty" data-testid="assembly-empty" />}
            {busy && program && <div className="compiling-overlay" />}
          </div>
        </section>
      </div>

      {(hasDiagnostics || output) && <section className="inspector">
        <div className="inspector-toolbar">{(['diagnostics', 'output'] as const).filter(tab => tab === 'diagnostics' ? hasDiagnostics : output).map(tab => <button key={tab} id={`tab-${tab}`} aria-expanded={panel === tab} aria-controls="inspector-content" onClick={() => setPanel(panel === tab ? null : tab)}>{t[tab]}<ChevronDown size={12} /></button>)}</div>
        {panel && <div id="inspector-content" className="inspector-content" role="region" aria-labelledby={`tab-${panel}`}>
          {panel === 'diagnostics' && <div className="diagnostic-list">
            {compilation?.error && <p className="error-heading">{errorTitle}</p>}
            {compilation?.diagnostics.map((diagnostic, index) => <div className="diagnostic" key={index}>{diagnostic.line && <button disabled={stale} className="source-link" onClick={() => sourceEditor.current?.reveal(diagnostic.line!)}><bdi>kernel.py:{diagnostic.line}</bdi><ArrowRight size={12} /></button>}<pre dir="ltr">{diagnostic.message}</pre></div>)}
            {summary?.diagnostics.map((message, index) => <pre dir="ltr" key={index}>{message}</pre>)}
            {engineError && <div><p>{t.engineError}</p><pre dir="ltr">{engineError}</pre></div>}
          </div>}
          {panel === 'output' && <pre className="compiler-output" dir="ltr">{output}</pre>}
        </div>}
      </section>}
    </main>
    <dialog ref={dialog} onCancel={() => setPendingExample(null)} className="example-dialog"><h2>{t.restoreTitle}</h2><p>{t.restoreDescription}</p><div className="dialog-buttons"><button className="secondary-button" onClick={() => setPendingExample(null)}>{t.keep}</button><button className="compile-button" onClick={() => pendingExample && loadExample(pendingExample)}>{t.restore}</button></div></dialog>
  </div>
}
