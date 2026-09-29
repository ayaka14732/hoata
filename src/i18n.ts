import type { Language } from './types'

const en = {
  language: 'Language', target: 'Target', compile: 'Compile', compiling: 'Compiling', cancel: 'Cancel',
  assembly: 'TPU assembly', example: 'Example', custom: 'Custom',
  clamp: 'Pallas · clamp', square: 'JAX · square', matmul: 'Pallas · matmul',
  downloadSource: 'Download Python source', downloadAssembly: 'Download TPU assembly',
  compileHint: 'Compile with Ctrl / ⌘ + Enter',
  stale: 'Recompile to update', failed: 'Compilation failed',
  cancelled: 'Compilation cancelled', timeout: 'Compilation timed out', busy: 'Compiler is busy',
  program: 'Program', function: 'Jump to function',
  search: 'Find in assembly', previousMatch: 'Previous match', nextMatch: 'Next match',
  diagnostics: 'Diagnostics', output: 'Compiler output',
  restore: 'Load example', restoreTitle: 'Load this example?', restoreDescription: 'This replaces the current source. Download it first if you want to keep a copy.',
  keep: 'Keep editing', dismiss: 'Dismiss', serverError: 'Cannot reach the local compiler. Start the Hoata server and try again.',
  engineError: 'Could not build the source index.',
  sourceLabel: 'Python source editor', assemblyLabel: 'TPU assembly viewer',
  resize: 'Resize source and assembly panels', retry: 'Retry',
}
type Messages = { [K in keyof typeof en]: string }
const zh: Messages = {
  language: '语言', target: '编译目标', compile: '编译', compiling: '正在编译', cancel: '取消',
  assembly: 'TPU 汇编', example: '示例', custom: '自定义',
  clamp: 'Pallas · 限幅', square: 'JAX · 平方', matmul: 'Pallas · 矩阵乘法',
  downloadSource: '下载 Python 源码', downloadAssembly: '下载 TPU 汇编',
  compileHint: '按 Ctrl / ⌘ + Enter 编译',
  stale: '重新编译以更新', failed: '编译失败',
  cancelled: '已取消编译', timeout: '编译超时', busy: '编译器正忙',
  program: '程序', function: '定位函数',
  search: '搜索汇编', previousMatch: '上一个匹配', nextMatch: '下一个匹配',
  diagnostics: '诊断', output: '编译器输出',
  restore: '载入示例', restoreTitle: '载入这个示例？', restoreDescription: '这会替换当前源码。如需保留，请先下载源码。',
  keep: '继续编辑', dismiss: '关闭', serverError: '无法连接本地编译器，请启动 Hoata 服务后重试。',
  engineError: '无法建立源码索引。',
  sourceLabel: 'Python 源码编辑器', assemblyLabel: 'TPU 汇编查看器',
  resize: '调整源码与汇编面板宽度', retry: '重试',
}
const he: Messages = {
  language: 'שפה', target: 'יעד', compile: 'הידור', compiling: 'מהדר', cancel: 'ביטול',
  assembly: 'אסמבלי TPU', example: 'דוגמה', custom: 'מותאם אישית',
  clamp: 'Pallas · הגבלת ערכים', square: 'JAX · ריבוע', matmul: 'Pallas · כפל מטריצות',
  downloadSource: 'הורדת קוד Python', downloadAssembly: 'הורדת אסמבלי TPU',
  compileHint: 'הידור באמצעות Ctrl / ⌘ + Enter',
  stale: 'יש להדר שוב לעדכון', failed: 'ההידור נכשל',
  cancelled: 'ההידור בוטל', timeout: 'זמן ההידור הסתיים', busy: 'הקומפיילר עסוק',
  program: 'תוכנית', function: 'מעבר לפונקציה',
  search: 'חיפוש באסמבלי', previousMatch: 'התאמה קודמת', nextMatch: 'התאמה הבאה',
  diagnostics: 'אבחון', output: 'פלט הקומפיילר',
  restore: 'טעינת דוגמה', restoreTitle: 'לטעון את הדוגמה?', restoreDescription: 'הפעולה תחליף את קוד המקור הנוכחי. אפשר להוריד אותו קודם כדי לשמור עותק.',
  keep: 'המשך עריכה', dismiss: 'סגירה', serverError: 'לא ניתן להתחבר לקומפיילר המקומי. יש להפעיל את שרת Hoata ולנסות שוב.',
  engineError: 'לא ניתן לבנות את אינדקס קוד המקור.',
  sourceLabel: 'עורך קוד Python', assemblyLabel: 'מציג אסמבלי TPU',
  resize: 'שינוי רוחב חלוניות הקוד והאסמבלי', retry: 'ניסיון חוזר',
}
export const messages: Record<Language, Messages> = { en, zh, he }
export function initialLanguage(): Language {
  let saved: string | null = null
  try { saved = localStorage.getItem('hoata.language') } catch { /* Storage may be disabled. */ }
  if (saved === 'en' || saved === 'zh' || saved === 'he') return saved
  for (const language of navigator.languages) {
    if (/^(he|iw)(-|$)/i.test(language)) return 'he'
    if (/^zh(-|$)/i.test(language)) return 'zh'
    if (/^en(-|$)/i.test(language)) return 'en'
  }
  return 'en'
}
