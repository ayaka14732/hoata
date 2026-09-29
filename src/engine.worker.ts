/// <reference lib="webworker" />
import init, { SourceIndex } from '../wasm/pkg/hoata_engine'
import wasmUrl from '../wasm/pkg/hoata_engine_bg.wasm?url'
const ready = init({ module_or_path: wasmUrl })
let index: SourceIndex | undefined
self.onmessage = async (event: MessageEvent<{ id: number; method: string; value: string | number }>) => {
  const { id, method, value } = event.data
  try {
    await ready
    let result: unknown
    switch (method) {
      case 'load': {
        const next = new SourceIndex(String(value))
        index?.free()
        index = next
        result = JSON.parse(index.summary())
        break
      }
      case 'source': result = JSON.parse(index!.select_source(Number(value))); break
      case 'assembly': result = JSON.parse(index!.select_assembly(Number(value))); break
      case 'search': result = JSON.parse(index!.search(String(value))); break
      default: throw new Error(`Unknown index operation: ${method}`)
    }
    self.postMessage({ id, result })
  } catch (error) {
    self.postMessage({ id, error: String(error) })
  }
}
