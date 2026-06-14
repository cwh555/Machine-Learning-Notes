import { existsSync, readdirSync } from "node:fs"
import path from "node:path"

const backgroundDir = path.join(process.cwd(), "quartz", "static", "background")
const allowedExtensions = new Set([".png", ".jpg", ".jpeg", ".webp"])

export type PrivateBackgroundFile = {
  name: string
}

export function getPrivateBackgroundFiles(): PrivateBackgroundFile[] {
  if (!existsSync(backgroundDir)) return []

  return readdirSync(backgroundDir, { withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .filter((name) => !name.startsWith(".") && !name.startsWith("._"))
    .filter((name) => allowedExtensions.has(path.extname(name).toLowerCase()))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
    .map((name) => ({ name }))
}
