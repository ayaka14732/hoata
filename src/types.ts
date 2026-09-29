export type Language = 'en' | 'zh' | 'he'
export type Target = 'tpu-v4-tc' | 'tpu-v6e-tc'
export type ExampleId = 'clamp' | 'square' | 'matmul'
export interface Config {
  versions: Record<string, string>
  targets: { id: Target; name: string }[]
  examples: { id: ExampleId; source: string }[]
  timeout_seconds: number
  source_limit: number
}
export interface SourceFrame {
  path: string
  line_start: number
  line_end: number
  col_start: number
  col_end: number
  function_name: string
}
export interface Origin {
  hlo_name: string
  hlo_module_name: string
  hlo_module_id: string
  llo_ordinal: number
  locations: { frames: SourceFrame[]; primitive: string; scope_stack: string[] }[]
}
export interface Instruction {
  pc: number
  slot: string
  line: number
  mnemonic: string
}
export interface Program {
  id: string
  target: Target
  assembly: string
  instructions: Instruction[]
  bundles: { pc: number; line: number }[]
  source_map: { module_name: string; diagnostics: string[]; [key: string]: unknown }
}
export interface Compilation {
  ok: boolean
  id?: string
  target?: Target
  programs: Program[]
  versions?: Record<string, string>
  diagnostics: { message: string; line?: number | null }[]
  error?: string
  elapsed_ms?: number
  traceback?: string
  logs?: { stdout: string; stderr: string }
}
export interface Summary {
  program_id: string
  bundle_count: number
  instruction_count: number
  mapped_count: number
  status: string
  diagnostics: string[]
  mappings: { line: number; assembly_lines: number[] }[]
  functions: { id: number; name: string; lines: number[] }[]
  preferred_line: number
  pc_lines: { line: number; pc: number }[]
}
export interface Selection {
  assembly_lines: number[]
  source_lines: number[]
  details: {
    instruction: Instruction
    source: {
      compiler_annotation: string
      annotation_locations: SourceFrame[]
      origins: Origin[]
      function_symbols: number[]
    } | null
  }[]
}
export const emptySelection: Selection = { assembly_lines: [], source_lines: [], details: [] }
