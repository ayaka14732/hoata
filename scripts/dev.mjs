import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
const children = [
  spawn('cargo', ['run', '--locked', '-p', 'hoata-server'], { stdio: 'inherit' }),
  spawn(process.execPath, [fileURLToPath(new URL('../node_modules/vite/bin/vite.js', import.meta.url)), '--host', '127.0.0.1'], { stdio: 'inherit' }),
]
let stopping = false
function stop(code = 0) {
  if (stopping) return
  stopping = true
  for (const child of children) child.kill('SIGTERM')
  process.exitCode = code
}
for (const child of children) {
  child.on('exit', code => stop(code ?? 1))
  child.on('error', error => { console.error(error.message); stop(1) })
}
process.on('SIGINT', () => stop())
process.on('SIGTERM', () => stop())
