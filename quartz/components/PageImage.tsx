import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path"
import { getPrivateCharacters } from "../util/privateCharacters"

const characterStorageKey = "private-character"
const characterChangeEventName = "private-character-change"

function normalizeImageName(imageName: unknown): string | undefined {
  if (typeof imageName !== "string") return undefined

  const normalized = imageName.trim().replace(/^\/+/, "").replace(/^images\/+/, "")
  return normalized.length > 0 ? normalized : undefined
}

const PageImage: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
  const imageName = normalizeImageName(fileData.frontmatter?.image)
  if (!imageName) return null

  const characters = getPrivateCharacters().map((character) => ({
    id: character.id,
    name: character.name,
    images: character.images.map((image) => ({
      name: image.name,
      path: image.path,
      src: joinSegments(pathToRoot(fileData.slug!), "static/characters", character.id, image.path),
    })),
  }))

  const fallbackImage = characters.flatMap((character) => character.images).find((image) => image.name === imageName)
  if (!fallbackImage) return null

  return (
    <div
      className={`page-image ${displayClass ?? ""}`}
      data-image-name={imageName}
      data-characters={JSON.stringify(characters)}
    >
      <img
        src={fallbackImage.src}
        alt={fileData.frontmatter?.title || "Featured Image"}
        style={{
          width: "100%",
          borderRadius: "8px",
          marginBottom: "1rem",
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

  const findImage = (character, imageName) => {
    if (!character || !Array.isArray(character.images)) return null
    return character.images.find((image) => image.name === imageName) ?? null
  }

  const updatePageImage = (root) => {
    if (!(root instanceof HTMLElement)) return

    const imageName = root.dataset.imageName
    const image = root.querySelector("img")
    if (!(image instanceof HTMLImageElement) || !imageName) return

    const characters = parseCharacters(root)
    const selectedCharacterId = safeGetStorage(CHARACTER_STORAGE_KEY)
    const selectedCharacter = selectedCharacterId
      ? characters.find((character) => character.id === selectedCharacterId)
      : null
    const selectedImage = findImage(selectedCharacter, imageName)
    const fallbackImage = characters.map((character) => findImage(character, imageName)).find(Boolean)
    const nextImage = selectedImage ?? fallbackImage ?? null

    if (!nextImage?.src) {
      root.hidden = true
      image.removeAttribute("src")
      return
    }

    root.hidden = false
    if (image.src !== new URL(nextImage.src, window.location.href).href) {
      image.src = nextImage.src
    }
  }

  const updateAllPageImages = () => {
    document.querySelectorAll(".page-image[data-image-name]").forEach(updatePageImage)
  }

  const bindListenersOnce = () => {
    if (state.bound) return
    state.bound = true
    document.addEventListener("nav", updateAllPageImages)
    document.addEventListener(CHARACTER_CHANGE_EVENT_NAME, updateAllPageImages)
  }

  bindListenersOnce()
  updateAllPageImages()
})()
`

export default (() => PageImage) satisfies QuartzComponentConstructor