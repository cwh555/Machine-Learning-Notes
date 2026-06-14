import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path"
import { getPrivateCharacters } from "../util/privateCharacters"

const characterStorageKey = "private-character"
const characterChangeEventName = "private-character-change"

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

const PageImage: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
  const imageSlot = normalizeImageSlot(fileData.frontmatter?.image)
  if (!imageSlot) return null

  const root = pathToRoot(fileData.slug!)

  const characters = getPrivateCharacters().map((character) => ({
    id: character.id,
    name: character.name,
    images: character.images.map((image, index) => ({
      slot: String(index + 1),
      name: image.name,
      path: image.path,
      src: joinSegments(root, "static/characters", character.id, image.path),
    })),
  }))

  const fallbackImage = characters
    .flatMap((character) => character.images)
    .find((image) => image.slot === imageSlot)

  if (!fallbackImage) return null

  return (
    <div
      className={`page-image ${displayClass ?? ""}`}
      data-image-slot={imageSlot}
      data-characters={JSON.stringify(characters)}
    >
      <img
        src={fallbackImage.src}
        alt={fileData.frontmatter?.title || "Featured Image"}
        style={{
          width: "100%",
          borderRadius: "8px",
          margin: "0 0 1rem",
          height: "auto",
          display: "block",
        }}
      />
    </div>
  )
}

PageImage.afterDOMLoaded = `
(() => {
  const CHARACTER_STORAGE_KEY = ${JSON.stringify(characterStorageKey)}
  const CHARACTER_CHANGE_EVENT_NAME = ${JSON.stringify(characterChangeEventName)}

  const state = window.__quartzPageImage ?? { bound: false }
  window.__quartzPageImage = state

  const safeGetStorage = (key) => {
    try {
      return localStorage.getItem(key)
    } catch {
      return null
    }
  }

  const parseCharacters = (root) => {
    try {
      const raw = root.dataset.characters ?? "[]"
      const parsed = JSON.parse(raw)
      return Array.isArray(parsed) ? parsed : []
    } catch {
      return []
    }
  }

  const findImageBySlot = (character, slot) => {
    if (!character || !Array.isArray(character.images) || character.images.length === 0) return null

    const exactImage = character.images.find((image) => image.slot === slot)
    if (exactImage) return exactImage

    if (/^\d+$/.test(slot)) {
      const slotNumber = Number.parseInt(slot, 10)
      if (Number.isFinite(slotNumber) && slotNumber > 0) {
        return character.images[(slotNumber - 1) % character.images.length] ?? null
      }
    }

    return null
  }

  const pickImageSrc = (characters, slot) => {
    const selectedCharacterId = safeGetStorage(CHARACTER_STORAGE_KEY)

    const selectedCharacter = selectedCharacterId
      ? characters.find((character) => character.id === selectedCharacterId)
      : null

    const selectedImage = findImageBySlot(selectedCharacter, slot)
    if (selectedImage?.src) return selectedImage.src

    for (const character of characters) {
      const fallbackImage = findImageBySlot(character, slot)
      if (fallbackImage?.src) return fallbackImage.src
    }

    return null
  }

  const updatePageImage = (root) => {
    if (!(root instanceof HTMLElement)) return

    const image = root.querySelector("img")
    if (!(image instanceof HTMLImageElement)) return

    const slot = root.dataset.imageSlot
    if (!slot) return

    const characters = parseCharacters(root)
    const nextSrc = pickImageSrc(characters, slot)

    if (!nextSrc) {
      root.hidden = true
      image.removeAttribute("src")
      return
    }

    root.hidden = false

    const resolvedNextSrc = new URL(nextSrc, window.location.href).href
    if (image.src !== resolvedNextSrc) {
      image.src = nextSrc
    }
  }

  const updateAllPageImages = () => {
    document.querySelectorAll(".page-image[data-image-slot]").forEach(updatePageImage)
  }

  const bindListenersOnce = () => {
    if (state.bound) return
    state.bound = true

    document.addEventListener("nav", updateAllPageImages)
    document.addEventListener(CHARACTER_CHANGE_EVENT_NAME, updateAllPageImages)

    window.addEventListener("storage", (event) => {
      if (event.key === CHARACTER_STORAGE_KEY) {
        updateAllPageImages()
      }
    })
  }

  bindListenersOnce()
  updateAllPageImages()
})()
`

export default (() => PageImage) satisfies QuartzComponentConstructor