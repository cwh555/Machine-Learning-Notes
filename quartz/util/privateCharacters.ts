import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs"
import path from "node:path"

const charactersDir = path.join(process.cwd(), "quartz", "static", "characters")
const characterConfigFileName = "character.json"
const backgroundDirName = "backgrounds"
const animationDirName = "animation"
const imageDirName = "images"
const allowedImageExtensions = new Set([".png", ".jpg", ".jpeg", ".webp"])
const allowedAnimationExtensions = new Set([".png"])

export type PrivateCharacterDefaults = {
  backgroundEnabled: boolean
  backgroundOpacity: number
  slideshowEnabled: boolean
  slideshowSeconds: number
  animationEnabled: boolean
  animationSize: number
  animationOpacity: number
  animationFps: number
  animationMinGuard: number
}

export type PrivateCharacterAsset = {
  name: string
  path: string
}

export type PrivateCharacterProfile = {
  id: string
  name: string
  defaults: PrivateCharacterDefaults
  backgrounds: PrivateCharacterAsset[]
  images: PrivateCharacterAsset[]
  animation: {
    frames: PrivateCharacterAsset[]
  }
  mark: PrivateCharacterAsset | null
}

type CharacterConfig = {
  id?: unknown
  name?: unknown
  defaults?: Partial<Record<keyof PrivateCharacterDefaults, unknown>>
}

const defaultCharacterDefaults: PrivateCharacterDefaults = {
  backgroundEnabled: true,
  backgroundOpacity: 15,
  slideshowEnabled: true,
  slideshowSeconds: 8,
  animationEnabled: true,
  animationSize: 100,
  animationOpacity: 100,
  animationFps: 8,
  animationMinGuard: 1200,
}

function isVisibleEntryName(name: string): boolean {
  return !name.startsWith(".") && !name.startsWith("._")
}

function toDisplayName(id: string): string {
  return id
    .replace(/[-_]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function sortFileNames(names: string[]): string[] {
  return [...names].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
}

function normalizeBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback
}

function normalizeNumber(value: unknown, fallback: number, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback
  return Math.min(max, Math.max(min, Math.round(value)))
}

function normalizeDefaults(defaults: CharacterConfig["defaults"]): PrivateCharacterDefaults {
  return {
    backgroundEnabled: normalizeBoolean(
      defaults?.backgroundEnabled,
      defaultCharacterDefaults.backgroundEnabled,
    ),
    backgroundOpacity: normalizeNumber(
      defaults?.backgroundOpacity,
      defaultCharacterDefaults.backgroundOpacity,
      0,
      100,
    ),
    slideshowEnabled: normalizeBoolean(
      defaults?.slideshowEnabled,
      defaultCharacterDefaults.slideshowEnabled,
    ),
    slideshowSeconds: normalizeNumber(
      defaults?.slideshowSeconds,
      defaultCharacterDefaults.slideshowSeconds,
      1,
      3600,
    ),
    animationEnabled: normalizeBoolean(
      defaults?.animationEnabled,
      defaultCharacterDefaults.animationEnabled,
    ),
    animationSize: normalizeNumber(
      defaults?.animationSize,
      defaultCharacterDefaults.animationSize,
      10,
      500,
    ),
    animationOpacity: normalizeNumber(
      defaults?.animationOpacity,
      defaultCharacterDefaults.animationOpacity,
      0,
      100,
    ),
    animationFps: normalizeNumber(
      defaults?.animationFps,
      defaultCharacterDefaults.animationFps,
      1,
      30,
    ),
    animationMinGuard: normalizeNumber(
      defaults?.animationMinGuard,
      defaultCharacterDefaults.animationMinGuard,
      0,
      5000,
    ),
  }
}

function readImageAssets(directory: string, relativeDirectory: string, allowedExtensions: Set<string>) {
  if (!existsSync(directory)) return []

  return sortFileNames(
    readdirSync(directory, { withFileTypes: true })
      .filter((entry) => entry.isFile())
      .map((entry) => entry.name)
      .filter(isVisibleEntryName)
      .filter((name) => allowedExtensions.has(path.extname(name).toLowerCase())),
  ).map((name) => ({
    name,
    path: path.posix.join(relativeDirectory, name),
  }))
}

function readMark(characterDir: string): PrivateCharacterAsset | null {
  const markName = sortFileNames(
    readdirSync(characterDir, { withFileTypes: true })
      .filter((entry) => entry.isFile())
      .map((entry) => entry.name)
      .filter(isVisibleEntryName)
      .filter((name) => path.parse(name).name === "mark")
      .filter((name) => allowedImageExtensions.has(path.extname(name).toLowerCase())),
  )[0]

  return markName ? { name: markName, path: markName } : null
}

function readConfig(configPath: string, fallbackId: string): CharacterConfig {
  if (!existsSync(configPath)) return {}

  try {
    const parsed = JSON.parse(readFileSync(configPath, "utf8")) as CharacterConfig
    return parsed && typeof parsed === "object" ? parsed : {}
  } catch (error) {
    console.warn(`[privateCharacters] Failed to parse ${configPath}; using defaults.`, error)
    return { id: fallbackId, name: toDisplayName(fallbackId), defaults: defaultCharacterDefaults }
  }
}

function ensureConfig(configPath: string, id: string): CharacterConfig {
  if (existsSync(configPath)) return readConfig(configPath, id)

  const generatedConfig = {
    id,
    name: toDisplayName(id),
    defaults: defaultCharacterDefaults,
  }

  try {
    mkdirSync(path.dirname(configPath), { recursive: true })
    writeFileSync(configPath, `${JSON.stringify(generatedConfig, null, 2)}\n`, "utf8")
  } catch (error) {
    console.warn(`[privateCharacters] Failed to create ${configPath}.`, error)
  }

  return generatedConfig
}

export function getPrivateCharacters(): PrivateCharacterProfile[] {
  if (!existsSync(charactersDir)) return []

  return readdirSync(charactersDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .filter(isVisibleEntryName)
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
    .map((id) => {
      const characterDir = path.join(charactersDir, id)
      const backgrounds = readImageAssets(
        path.join(characterDir, backgroundDirName),
        backgroundDirName,
        allowedImageExtensions,
      )
      const frames = readImageAssets(
        path.join(characterDir, animationDirName),
        animationDirName,
        allowedAnimationExtensions,
      )
      const images = readImageAssets(
        path.join(characterDir, imageDirName),
        imageDirName,
        allowedImageExtensions,
      )
      const mark = readMark(characterDir)

      if (backgrounds.length === 0 && frames.length === 0 && images.length === 0 && !mark) return null

      const config = ensureConfig(path.join(characterDir, characterConfigFileName), id)
      const name = typeof config.name === "string" && config.name.trim() ? config.name.trim() : toDisplayName(id)

      return {
        id,
        name,
        defaults: normalizeDefaults(config.defaults),
        backgrounds,
        images,
        animation: { frames },
        mark,
      } satisfies PrivateCharacterProfile
    })
    .filter((character): character is PrivateCharacterProfile => character !== null)
}
