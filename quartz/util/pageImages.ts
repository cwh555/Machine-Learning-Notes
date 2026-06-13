import path from "path"
import { readdir } from "fs/promises"
import { QUARTZ, joinSegments } from "./path"
import type { BuildCtx } from "./ctx"
import type { ProcessedContent } from "../plugins/vfile"

const imageExtensions = new Set([".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"])
const imageLabel = "image"

function normalizeImageName(imageName: unknown): string | undefined {
  if (typeof imageName !== "string") return undefined

  const normalized = imageName.trim().replace(/^\/+/, "")
  return normalized.length > 0 ? normalized : undefined
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

function hasImageLabel(frontmatter: Record<string, unknown>): boolean {
  // Support both a single label and a list of labels so authors can choose the lighter syntax.
  const labels = [
    ...coerceToStringArray(frontmatter.label),
    ...coerceToStringArray(frontmatter.labels),
    ...coerceToStringArray(frontmatter.tags),
  ]

  return labels.some((label) => label.toLowerCase() === imageLabel)
}

async function listAvailableImages(): Promise<string[]> {
  const imageDir = joinSegments(QUARTZ, "static", "images")

  let entries: string[]
  try {
    entries = await readdir(imageDir)
  } catch (error) {
    console.warn(`[page-images] unable to read ${imageDir}: ${(error as Error).message}`)
    return []
  }

  return entries
    .filter((entry) => !entry.startsWith(".") && !entry.startsWith("._"))
    .filter((entry) => imageExtensions.has(path.extname(entry).toLowerCase()))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
}

export async function assignPageImages(_ctx: BuildCtx, content: ProcessedContent[]): Promise<void> {
  const availableImages = await listAvailableImages()
  if (availableImages.length === 0) return

  const usedImages = new Set<string>()
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

    const manualImage = normalizeImageName(frontmatter.image)
    if (manualImage) {
      // Manual images always win; reserve them so auto assignment does not reuse the same file.
      usedImages.add(manualImage)
      frontmatter.image = manualImage
      continue
    }

    if (hasImageLabel(frontmatter)) {
      pagesNeedingImages.push(processedContent)
    }
  }

  let imageIndex = 0
  for (const processedContent of pagesNeedingImages) {
    const file = processedContent[1]
    const frontmatter = file.data.frontmatter as Record<string, unknown> | undefined
    if (!frontmatter) continue

    while (imageIndex < availableImages.length && usedImages.has(availableImages[imageIndex])) {
      imageIndex++
    }

    const imageName = availableImages[imageIndex]
    if (!imageName) {
      console.warn(
        `[page-images] no unused image left for ${
          file.data.relativePath ?? file.data.slug ?? "unknown file"
        }`,
      )
      continue
    }

    frontmatter.image = imageName
    usedImages.add(imageName)
    imageIndex++
  }
}
