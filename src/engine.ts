export class Engine {
  private worker = new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' })
  private serial = 0
  private requests = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>()
  constructor() {
    this.worker.onmessage = ({ data }: MessageEvent<{ id: number; result: unknown; error?: string }>) => {
      const request = this.requests.get(data.id)
      if (!request) return
      this.requests.delete(data.id)
      if (data.error) request.reject(new Error(data.error))
      else request.resolve(data.result)
    }
    this.worker.onerror = event => {
      for (const request of this.requests.values()) request.reject(new Error(event.message))
      this.requests.clear()
    }
  }
  call<T>(method: string, value: string | number): Promise<T> {
    return new Promise((resolve, reject) => {
      const id = ++this.serial
      this.requests.set(id, { resolve: value => resolve(value as T), reject })
      this.worker.postMessage({ id, method, value })
    })
  }
  dispose() {
    this.worker.terminate()
    for (const request of this.requests.values()) request.reject(new Error('Index closed'))
    this.requests.clear()
  }
}
