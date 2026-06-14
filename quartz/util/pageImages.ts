import { getPrivateCharacters } from "./privateCharacters"
import type { BuildCtx } from "./ctx"
import type { ProcessedContent } from "../plugins/vfile"

const imageProperty = "image"

function normalizeImageSlot(imageName: unknown): string | undefined {
  if (typeof imageName !== "string" && typeof imageName !== "number") return undefined

  const normalized = imageName
    .toString()
    .trim()
    .replace(/^\/+/, "")
    .replace(/^images\/+/, "")

  if (normalized.length === 0) return undefined

  const basename = normalized.replace(/\.[^.\/]+$/, "")

  if (/^\d+$/.test(basename)) {
    return String(Number.parseInt(basename, 10))
  }

  return basename
}

function coerceToStringArray(input: unknown): string[] {
  if (input === undefined || input === null) return []

  const values = Array.isArray(input) ? input : input.toString().split(",")
  return values
    .filter(
      (value): value is string | number =>
        typeof value === "string" || typeof value === "number",
    )
    .map((value) => value.toString().trim())
    .filter((value) => value.length > 0)
}

function hasImageProperty(frontmatter: Record<string, unknown>): boolean {
  const properties = [
    ...coerceToStringArray(frontmatter.property),
    ...coerceToStringArray(frontmatter.properties),
  ]

  return properties.some((property) => property.toLowerCase() === imageProperty)
}

function listAvailableImageSlots(): string[] {
  const imageCounts = getPrivateCharacters()
    .map((character) => character.images.length)
    .filter((count) => count > 0)

  if (imageCounts.length === 0) {
    console.warn("[page-images] no character images found under quartz/static/characters/<id>/images")
    return []
  }

  // 取所有角色中最多的圖片數。
  // 每篇 properties: image 的 note 都應該拿到 slot；
  // 如果某個角色圖片較少，交給 PageImage 依該角色圖片數循環 fallback。
  const availableCount = Math.max(...imageCounts)

  return Array.from({ length: availableCount }, (_, index) => String(index + 1))
}

export async function assignPageImages(_ctx: BuildCtx, content: ProcessedContent[]): Promise<void> {
  const availableSlots = listAvailableImageSlots()
  if (availableSlots.length === 0) return

  const usedSlots = new Set<string>()
  const pagesNeedingImages: ProcessedContent[] = []

  const sortedContent = [...content].sort((a, b) => {
    const aFile = a[1]
    const bFile = b[1]
    const aKey = aFile.data.slug ?? aFile.data.relativePath ?? ""
    const bKey = bFile.data.slug ?? bFile.data.relativePath ?? ""
    return aKey.toString().localeCompare(bKey.toString())
  })

  for (const processedContent of sortedContent) {
    const file = processedContent[1]
    const frontmatter = file.data.frontmatter as Record<string, unknown> | undefined
    if (!frontmatter) continue

    const existingImageSlot = normalizeImageSlot(frontmatter.image)
    if (existingImageSlot) {
      usedSlots.add(existingImageSlot)
      frontmatter.image = existingImageSlot
      continue
    }

    if (hasImageProperty(frontmatter)) {
      pagesNeedingImages.push(processedContent)
    }
  }

  let slotIndex = 0

  for (const processedContent of pagesNeedingImages) {
    const file = processedContent[1]
    const frontmatter = file.data.frontmatter as Record<string, unknown> | undefined
    if (!frontmatter) continue

    while (slotIndex < availableSlots.length && usedSlots.has(availableSlots[slotIndex])) {
      slotIndex++
    }

    const imageSlot = availableSlots[slotIndex]
    if (!imageSlot) {
      console.warn(
        `[page-images] no unused image slot left for ${
          file.data.relativePath ?? file.data.slug ?? "unknown file"
        }`,
      )
      continue
    }

    frontmatter.image = imageSlot
    usedSlots.add(imageSlot)
    slotIndex++
  }
}