import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import * as monaco from 'monaco-editor/editor/editor.api.js'
import 'monaco-editor/languages/definitions/python/register.js'
import EditorWorker from 'monaco-editor/editor/editor.worker.js?worker'
import type { Summary } from './types'

self.MonacoEnvironment = { getWorker: () => new EditorWorker() }
monaco.languages.register({ id: 'tpuasm' })
// Slot/register vocabulary follows vscode-tpuasm; executable bytes and parsed
// instruction coordinates are supplied by tpuasm itself.
monaco.languages.setMonarchTokensProvider('tpuasm', {
  tokenizer: {
    root: [
      [/#.*$/, 'comment'],
      [/\.(?:target|encoding|align|empty)\b/, 'keyword'],
      [/\b(?:s[01]|dma|va[0-3]|vst|vld[01]?|cld|vx[01]|vr[01]|misc)(?=:)/, 'type'],
      [/\b(?:entry|L_[0-9a-f]+)(?=:|\b)/, 'tag'],
      [/@!?p\d+/, 'keyword'],
      [/\b(?:s|v|p|a|gmr|gsfn|gsft|acc|r)\d+\b/, 'variable'],
      [/\b(?:smem|vmem|cmem|hbm|sflag)\b/, 'type'],
      [/-?(?:0x[0-9a-f]+|\d+(?:\.\d+)?(?:e[-+]?\d+)?)\b/i, 'number'],
      [/\b[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*\b/, 'identifier'],
      [/[{}\[\]():;,=]/, 'delimiter'],
    ],
  },
})
monaco.editor.defineTheme('hoata', {
  base: 'vs-dark', inherit: true,
  rules: [
    { token: 'comment', foreground: '737C79' },
    { token: 'keyword', foreground: 'C89ACD' },
    { token: 'string', foreground: 'B3C98C' },
    { token: 'number', foreground: 'D6B781' },
    { token: 'type', foreground: '78BBB8' },
    { token: 'variable', foreground: 'B3C8DB' },
    { token: 'identifier', foreground: 'D5DADE' },
    { token: 'tag', foreground: 'D6B781' },
  ],
  colors: {
    'editor.background': '#18191b', 'editor.foreground': '#d5dade',
    'editorLineNumber.foreground': '#55585e', 'editorLineNumber.activeForeground': '#b4b7be',
    'editor.lineHighlightBackground': '#ffffff04', 'editor.lineHighlightBorder': '#00000000',
    'editor.selectionBackground': '#65738355', 'editor.inactiveSelectionBackground': '#65738330',
    'editorCursor.foreground': '#e7e9ec', 'editorIndentGuide.background1': '#26282c',
    'editorIndentGuide.activeBackground1': '#41444b', 'editorGutter.background': '#18191b',
    'scrollbarSlider.background': '#666a7133', 'scrollbarSlider.hoverBackground': '#666a7166',
    'editorOverviewRuler.border': '#00000000',
  },
})

export interface EditorHandle {
  reveal: (line: number) => void
  decorate: (summary: Summary | null, selected: number[], matches?: number[]) => void
  markErrors: (errors: { message: string; line?: number | null }[]) => void
  focus: () => void
}
interface Props {
  kind: 'source' | 'assembly'
  value: string
  label: string
  onChange?: (value: string) => void
  onSelect: (line: number) => void
  onCompile: () => void
  onCursor?: (line: number, column: number) => void
}

export const CodeEditor = forwardRef<EditorHandle, Props>(function CodeEditor(props, ref) {
  const container = useRef<HTMLDivElement>(null)
  const editor = useRef<monaco.editor.IStandaloneCodeEditor | null>(null)
  const decorations = useRef<monaco.editor.IEditorDecorationsCollection | null>(null)
  const current = useRef(props)
  current.current = props
  const replacing = useRef(false)
  const pcLines = useRef(new Map<number, number>())

  useImperativeHandle(ref, () => ({
    reveal(line) {
      editor.current?.revealLineInCenterIfOutsideViewport(line, monaco.editor.ScrollType.Smooth)
      editor.current?.setPosition({ lineNumber: line, column: 1 }, 'hoata')
    },
    focus() { editor.current?.focus() },
    markErrors(errors) {
      const model = editor.current?.getModel()
      if (model) monaco.editor.setModelMarkers(model, 'hoata', errors.filter(error => error.line).map(error => ({
        severity: monaco.MarkerSeverity.Error, message: error.message,
        startLineNumber: error.line!, endLineNumber: error.line!, startColumn: 1, endColumn: model.getLineMaxColumn(error.line!),
      })))
    },
    decorate(summary, selected, matches = []) {
      const items: monaco.editor.IModelDeltaDecoration[] = []
      const colors = new Map<number, number>()
      for (const mapping of summary?.mappings ?? []) {
        for (const line of props.kind === 'source' ? [mapping.line] : mapping.assembly_lines) {
          if (!colors.has(line)) colors.set(line, mapping.line % 6)
        }
      }
      for (const [line, color] of colors) items.push({ range: new monaco.Range(line, 1, line, 1), options: {
        isWholeLine: true, className: `mapped-line map-${color}`, linesDecorationsClassName: `mapping-stripe stripe-${color}`,
      } })
      for (const line of selected) items.push({ range: new monaco.Range(line, 1, line, 1), options: {
        isWholeLine: true, className: 'selected-source-line', linesDecorationsClassName: 'selected-stripe', zIndex: 10,
      } })
      for (const line of matches) items.push({ range: new monaco.Range(line, 1, line, 1), options: {
        isWholeLine: true, className: 'search-line', zIndex: 5,
      } })
      decorations.current?.set(items)
      pcLines.current = new Map(summary?.pc_lines.map(item => [item.line, item.pc]) ?? [])
      if (props.kind === 'assembly') editor.current?.updateOptions({ lineNumbers: line => {
        const pc = pcLines.current.get(line)
        return pc === undefined ? '' : pc.toString(16).padStart(4, '0')
      } })
    },
  }), [props.kind])

  useEffect(() => {
    const instance = monaco.editor.create(container.current!, {
      value: current.current.value,
      language: props.kind === 'source' ? 'python' : 'tpuasm', theme: 'hoata',
      readOnly: props.kind === 'assembly', domReadOnly: props.kind === 'assembly',
      automaticLayout: true, minimap: { enabled: false }, fontSize: 12.5, lineHeight: 22,
      fontFamily: '"JetBrains Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace',
      fontLigatures: false, fontWeight: '400', padding: { top: 18, bottom: 24 },
      scrollBeyondLastLine: false, renderLineHighlight: 'line', overviewRulerLanes: 0,
      hideCursorInOverviewRuler: true, folding: props.kind === 'source',
      lineNumbersMinChars: props.kind === 'assembly' ? 5 : 3, lineDecorationsWidth: 12,
      glyphMargin: false, contextmenu: false, stickyScroll: { enabled: false },
      wordWrap: 'off', tabSize: 4, insertSpaces: true, unicodeHighlight: { ambiguousCharacters: false, nonBasicASCII: false },
      ariaLabel: current.current.label, links: false,
      scrollbar: { verticalScrollbarSize: 8, horizontalScrollbarSize: 8, alwaysConsumeMouseWheel: false },
    })
    editor.current = instance
    decorations.current = instance.createDecorationsCollection()
    const change = instance.onDidChangeModelContent(() => {
      if (!replacing.current) current.current.onChange?.(instance.getValue())
    })
    const position = instance.onDidChangeCursorPosition(event => {
      current.current.onCursor?.(event.position.lineNumber, event.position.column)
      if (event.source !== 'hoata' && !replacing.current) current.current.onSelect(event.position.lineNumber)
    })
    const click = instance.onMouseDown(event => {
      if (event.target.position) current.current.onSelect(event.target.position.lineNumber)
    })
    const action = instance.addAction({ id: 'hoata.compile', label: 'Compile', keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter], run: () => current.current.onCompile() })
    return () => {
      change.dispose(); position.dispose(); click.dispose(); action.dispose()
      const model = instance.getModel()
      instance.dispose(); model?.dispose(); editor.current = null
    }
  }, [props.kind])

  useEffect(() => {
    const instance = editor.current
    if (instance && props.value !== instance.getValue()) {
      replacing.current = true
      instance.setValue(props.value)
      replacing.current = false
    }
  }, [props.value])
  useEffect(() => { editor.current?.updateOptions({ ariaLabel: props.label }) }, [props.label])
  return <div ref={container} className="code-editor" dir="ltr" data-editor={props.kind} />
})
