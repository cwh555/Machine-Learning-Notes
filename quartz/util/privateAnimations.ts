import { existsSync, readdirSync } from "node:fs"
import path from "node:path"

const animationDir = path.join(process.cwd(), "quartz", "static", "animation")
const allowedExtension = ".png"

export type PrivateAnimationFrame = {
  name: string
}

export type PrivateAnimationCharacter = {
  name: string
  frames: PrivateAnimationFrame[]
}

function parseFrameIndex(characterName: string, fileName: string): number | null {
  const escapedName = characterName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  const match = fileName.match(new RegExp(`^${escapedName}_(\\d+)\\.png$`, "i"))
  if (!match) return null
  return Number.parseInt(match[1], 10)
}

export function getPrivateAnimationCharacters(): PrivateAnimationCharacter[] {
  if (!existsSync(animationDir)) return []

  return readdirSync(animationDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .filter((name) => !name.startsWith(".") && !name.startsWith("._"))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
    .map((characterName) => {
      const characterDir = path.join(animationDir, characterName)
      const frames = readdirSync(characterDir, { withFileTypes: true })
        .filter((entry) => entry.isFile())
        .map((entry) => entry.name)
        .filter((name) => !name.startsWith(".") && !name.startsWith("._"))
        .filter((name) => path.extname(name).toLowerCase() === allowedExtension)
        .map((name) => ({ name, index: parseFrameIndex(characterName, name) }))
        .filter((frame): frame is { name: string; index: number } => frame.index !== null)
        .sort((a, b) => a.index - b.index || a.name.localeCompare(b.name, undefined, { numeric: true }))
        .map(({ name }) => ({ name }))

      return { name: characterName, frames }
    })
    .filter((character) => character.frames.length > 0)
}
