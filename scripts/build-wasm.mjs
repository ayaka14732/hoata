import { execFileSync } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
const root = fileURLToPath(new URL('../', import.meta.url))
const run = (command, args) => execFileSync(command, args, { cwd: root, stdio: 'inherit' })
run('cargo', ['build', '-p', 'hoata-engine', '--release', '--target', 'wasm32-unknown-unknown', '--locked'])
mkdirSync(`${root}/wasm/pkg`, { recursive: true })
run('wasm-bindgen', [`${root}/target/wasm32-unknown-unknown/release/hoata_engine.wasm`, '--target', 'web', '--out-dir', `${root}/wasm/pkg`, '--out-name', 'hoata_engine'])
