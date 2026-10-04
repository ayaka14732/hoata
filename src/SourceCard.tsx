import { CornerDownRight, X } from 'lucide-react'
import type { Messages } from './i18n'
import type { Selection, SourceChain, SourceFrame } from './types'

interface Props {
  details: Selection['details']
  source: string
  t: Messages
  onReveal: (line: number) => void
  onClose: () => void
  onHover: (active: boolean) => void
}

// The source text of a frame, with the recorded column range marked. Frames
// outside kernel.py (or beyond the compiled text) show only their coordinates.
function Snippet({ frame, lines }: { frame: SourceFrame; lines: string[] }) {
  const text = frame.path === 'kernel.py' ? lines[frame.line_start - 1] : undefined
  if (text === undefined) return <span className="source-snippet" />
  const end = frame.line_end === frame.line_start ? frame.col_end : text.length
  const indent = text.length - text.trimStart().length
  const start = Math.max(frame.col_start, indent)
  return <span className="source-snippet">
    {text.slice(indent, start)}<mark>{text.slice(start, end)}</mark>{text.slice(end)}{frame.line_end > frame.line_start ? ' …' : ''}
  </span>
}

function coordinate(frame: SourceFrame) {
  const path = frame.path === 'kernel.py' ? '' : `${frame.path}:`
  const end = frame.line_end === frame.line_start ? frame.col_end : `${frame.line_end}:${frame.col_end}`
  return `${path}${frame.line_start}:${frame.col_start}–${end}`
}

function labels(chain: SourceChain) {
  return chain.labels.map(label => `${label.scopes.length ? `${label.scopes.join(', ')} · ` : ''}LLO ${label.ordinals.join(', ')}`).join('; ')
}

function FrameRow({ frame, lines, className, tag, onReveal }: { frame: SourceFrame; lines: string[]; className: string; tag?: string; onReveal: (line: number) => void }) {
  return <button className={`source-row ${className}`} onClick={() => onReveal(frame.line_start)} disabled={frame.path !== 'kernel.py'}>
    <bdi className="source-coordinate">{coordinate(frame)}</bdi>
    <Snippet frame={frame} lines={lines} />
    {tag && <span className="source-tag" dir="ltr">{tag}</span>}
  </button>
}

// Structured replacement for the listing's `inner <- caller [scope; LLO n] | loc`
// comment: one block per chain, callers indented beneath the innermost frame,
// native locations and notes on their own labelled rows.
export function SourceCard({ details, source, t, onReveal, onClose, onHover }: Props) {
  const lines = source.split('\n')
  return <div className="source-card" role="region" aria-label={t.instructionSource} onMouseEnter={() => onHover(true)} onMouseLeave={() => onHover(false)}>
    <div className="source-card-header"><span>{t.instructionSource}</span><button className="icon-button" onClick={onClose} aria-label={t.dismiss} title={t.dismiss}><X size={12} /></button></div>
    <div className="source-card-body" dir="ltr">
      {details.map(({ instruction, view }) => <section key={instruction.line} className="source-entry">
        <div className="source-instruction"><span className="source-slot">{instruction.slot}</span><code>{instruction.mnemonic}</code></div>
        {view.chains.map((chain, index) => <div key={index} className="source-chain">
          {chain.frames.length
            ? chain.frames.map((frame, depth) => depth === 0
              ? <FrameRow key={depth} frame={frame} lines={lines} className="primary" tag={labels(chain)} onReveal={onReveal} />
              : <div key={depth} className="caller-row" style={{ paddingInlineStart: `${depth * 14}px` }}>
                <CornerDownRight size={11} aria-label={t.calledFrom} />
                <FrameRow frame={frame} lines={lines} className="caller" onReveal={onReveal} />
              </div>)
            : <div className="source-row unlocated"><span className="source-coordinate">{t.noLocation}</span><span className="source-tag" dir="ltr">{labels(chain)}</span></div>}
        </div>)}
        {view.compiler.map((frame, index) => <FrameRow key={index} frame={frame} lines={lines} className="compiler" tag={t.compilerLocation} onReveal={onReveal} />)}
        {view.notes.map((note, index) => <div key={index} className="source-note"><span>{t.compilerNote}</span><code>{note}</code></div>)}
        {!view.chains.length && !view.compiler.length && <div className="source-unknown">{t.noSource}</div>}
      </section>)}
    </div>
  </div>
}
