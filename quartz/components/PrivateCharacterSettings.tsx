import styles from "./styles/privateCharacterSettings.scss"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path"
import { getPrivateCharacters } from "../util/privateCharacters"

const characterStorageKey = "private-character"
const oldAnimationCharacterStorageKey = "private-animation-character"
const backgroundEnabledStorageKey = "private-background-enabled"
const backgroundStorageKey = "private-background"
const backgroundOpacityStorageKey = "private-background-opacity"
const slideshowEnabledStorageKey = "private-background-slideshow-enabled"
const slideshowSecondsStorageKey = "private-background-slideshow-seconds"
const animationEnabledStorageKey = "private-animation-enabled"
const animationSizeStorageKey = "private-animation-size"
const animationOpacityStorageKey = "private-animation-opacity"
const animationFpsStorageKey = "private-animation-fps"
const animationMinGuardStorageKey = "private-animation-min-guard"
const characterChangeEventName = "private-character-change"
const backgroundChangeEventName = "private-background-change"
const animationChangeEventName = "private-animation-change"

const PrivateCharacterSettings: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const characters = getPrivateCharacters().map((character) => {
    const backgrounds = character.backgrounds.map((background) => ({
      name: background.name,
      path: background.path,
      src: joinSegments(pathToRoot(fileData.slug!), "static/characters", character.id, background.path),
      kind: "background",
    }))
    const images = character.images.map((image) => ({
      name: image.name,
      path: image.path,
      src: joinSegments(pathToRoot(fileData.slug!), "static/characters", character.id, image.path),
      kind: "image",
    }))
    const frames = character.animation.frames.map((frame) => ({
      name: frame.name,
      path: frame.path,
      src: joinSegments(pathToRoot(fileData.slug!), "static/characters", character.id, frame.path),
      kind: "animation",
    }))
    const preview = [...backgrounds.slice(0, 2), ...images.slice(0, 1), ...frames.slice(0, 1)].slice(0, 4)

    return {
      id: character.id,
      name: character.name,
      defaults: character.defaults,
      backgrounds,
      images,
      frames,
      mark: character.mark
        ? {
            name: character.mark.name,
            path: character.mark.path,
            src: joinSegments(pathToRoot(fileData.slug!), "static/characters", character.id, character.mark.path),
          }
        : null,
      preview,
    }
  })

  return (
    <section class="private-character-settings" data-characters={JSON.stringify(characters)}>
      <div class="private-character-header">
        <div>
          <h2 class="private-character-title">Private Character</h2>
          <p class="private-character-description">
            Select a character first. Background, animation, and mark settings are scoped to the selected character.
          </p>
        </div>
        <div class="private-character-selected">
          <span class="private-character-selected-label">selected</span>
          <span id="private-character-current" class="private-character-selected-name">
            none
          </span>
        </div>
      </div>

      <div id="private-character-list" class="private-character-list" />
    </section>
  )
}

PrivateCharacterSettings.afterDOMLoaded = `
(() => {
  const CHARACTER_STORAGE_KEY = ${JSON.stringify(characterStorageKey)}
  const OLD_ANIMATION_CHARACTER_STORAGE_KEY = ${JSON.stringify(oldAnimationCharacterStorageKey)}
  const BACKGROUND_ENABLED_STORAGE_KEY = ${JSON.stringify(backgroundEnabledStorageKey)}
  const BACKGROUND_STORAGE_KEY = ${JSON.stringify(backgroundStorageKey)}
  const BACKGROUND_OPACITY_STORAGE_KEY = ${JSON.stringify(backgroundOpacityStorageKey)}
  const SLIDESHOW_ENABLED_STORAGE_KEY = ${JSON.stringify(slideshowEnabledStorageKey)}
  const SLIDESHOW_SECONDS_STORAGE_KEY = ${JSON.stringify(slideshowSecondsStorageKey)}
  const ANIMATION_ENABLED_STORAGE_KEY = ${JSON.stringify(animationEnabledStorageKey)}
  const ANIMATION_SIZE_STORAGE_KEY = ${JSON.stringify(animationSizeStorageKey)}
  const ANIMATION_OPACITY_STORAGE_KEY = ${JSON.stringify(animationOpacityStorageKey)}
  const ANIMATION_FPS_STORAGE_KEY = ${JSON.stringify(animationFpsStorageKey)}
  const ANIMATION_MIN_GUARD_STORAGE_KEY = ${JSON.stringify(animationMinGuardStorageKey)}
  const CHARACTER_CHANGE_EVENT_NAME = ${JSON.stringify(characterChangeEventName)}
  const BACKGROUND_CHANGE_EVENT_NAME = ${JSON.stringify(backgroundChangeEventName)}
  const ANIMATION_CHANGE_EVENT_NAME = ${JSON.stringify(animationChangeEventName)}

  const getRoot = () => document.querySelector(".private-character-settings")
  const getList = () => document.getElementById("private-character-list")
  const getCurrent = () => document.getElementById("private-character-current")

  const safeGetStorage = (key) => {
    try {
      return localStorage.getItem(key)
    } catch {
      return null
    }
  }

  const safeSetStorage = (key, value) => {
    try {
      localStorage.setItem(key, value)
    } catch {}
  }

  const safeRemoveStorage = (key) => {
    try {
      localStorage.removeItem(key)
    } catch {}
  }

  const parseCharacters = () => {
    const root = getRoot()
    if (!(root instanceof HTMLElement)) return []

    try {
      const raw = root.dataset.characters ?? "[]"
      const parsed = JSON.parse(raw)
      return Array.isArray(parsed) ? parsed : []
    } catch {
      return []
    }
  }

  const getSelectedCharacter = () => {
    const id = safeGetStorage(CHARACTER_STORAGE_KEY)
    if (!id) return null
    return parseCharacters().find((character) => character.id === id) ?? null
  }

  const emitProfileChange = (character) => {
    document.dispatchEvent(new CustomEvent(CHARACTER_CHANGE_EVENT_NAME, { detail: { id: character.id, name: character.name } }))
    document.dispatchEvent(new CustomEvent(BACKGROUND_CHANGE_EVENT_NAME, { detail: { character: character.id, source: "character" } }))
    document.dispatchEvent(new CustomEvent(ANIMATION_CHANGE_EVENT_NAME, { detail: { character: character.id, source: "character" } }))
  }

  const applyCharacter = (character) => {
    const defaults = character.defaults ?? {}
    const firstBackground = character.backgrounds?.[0]
    const hasBackgrounds = Array.isArray(character.backgrounds) && character.backgrounds.length > 0
    const hasFrames = Array.isArray(character.frames) && character.frames.length > 0

    safeSetStorage(CHARACTER_STORAGE_KEY, character.id)
    safeRemoveStorage(OLD_ANIMATION_CHARACTER_STORAGE_KEY)

    safeSetStorage(BACKGROUND_ENABLED_STORAGE_KEY, defaults.backgroundEnabled && hasBackgrounds ? "true" : "false")
    if (firstBackground?.path) {
      safeSetStorage(BACKGROUND_STORAGE_KEY, firstBackground.path)
    } else {
      safeRemoveStorage(BACKGROUND_STORAGE_KEY)
    }
    safeSetStorage(BACKGROUND_OPACITY_STORAGE_KEY, String(defaults.backgroundOpacity ?? 15))
    safeSetStorage(SLIDESHOW_ENABLED_STORAGE_KEY, defaults.slideshowEnabled && hasBackgrounds ? "true" : "false")
    safeSetStorage(SLIDESHOW_SECONDS_STORAGE_KEY, String(defaults.slideshowSeconds ?? 8))

    safeSetStorage(ANIMATION_ENABLED_STORAGE_KEY, defaults.animationEnabled && hasFrames ? "true" : "false")
    safeSetStorage(ANIMATION_SIZE_STORAGE_KEY, String(defaults.animationSize ?? 100))
    safeSetStorage(ANIMATION_OPACITY_STORAGE_KEY, String(defaults.animationOpacity ?? 100))
    safeSetStorage(ANIMATION_FPS_STORAGE_KEY, String(defaults.animationFps ?? 8))
    safeSetStorage(ANIMATION_MIN_GUARD_STORAGE_KEY, String(defaults.animationMinGuard ?? 1200))

    updateSelection()
    renderCharacters()
    emitProfileChange(character)
  }

  const renderEmptyState = (list) => {
    const empty = document.createElement("p")
    empty.className = "private-character-empty"
    empty.textContent = "No private characters found."
    list.appendChild(empty)
  }

  const renderPreview = (card, character) => {
    const preview = document.createElement("span")
    preview.className = "private-character-preview"

    const images = Array.isArray(character.preview) ? character.preview : []
    if (images.length === 0) {
      preview.classList.add("empty")
      preview.textContent = character.name.slice(0, 1).toUpperCase()
      card.appendChild(preview)
      return
    }

    for (const item of images) {
      const image = document.createElement("img")
      image.src = item.src
      image.alt = ""
      image.loading = "lazy"
      image.setAttribute("aria-hidden", "true")
      image.className =
        item.kind === "animation" ? "is-animation" : item.kind === "image" ? "is-image" : "is-background"
      preview.appendChild(image)
    }

    card.appendChild(preview)
  }

  const renderMark = (meta, character) => {
    const mark = document.createElement("span")
    mark.className = "private-character-mark-preview"

    if (character.mark?.src) {
      const image = document.createElement("img")
      image.src = character.mark.src
      image.alt = ""
      image.loading = "lazy"
      image.setAttribute("aria-hidden", "true")
      mark.appendChild(image)
    } else {
      mark.textContent = "no mark"
    }

    meta.appendChild(mark)
  }

  const renderCharacters = () => {
    const list = getList()
    if (!list) return

    list.textContent = ""
    const characters = parseCharacters()
    if (characters.length === 0) {
      renderEmptyState(list)
      return
    }

    const selected = safeGetStorage(CHARACTER_STORAGE_KEY)
    for (const character of characters) {
      const card = document.createElement("button")
      card.type = "button"
      card.className = "private-character-card"
      if (character.id === selected) card.classList.add("active")

      renderPreview(card, character)

      const content = document.createElement("span")
      content.className = "private-character-card-content"

      const name = document.createElement("span")
      name.className = "private-character-card-name"
      name.textContent = character.name
      content.appendChild(name)

      const counts = document.createElement("span")
      counts.className = "private-character-card-counts"
      counts.textContent =
        (character.backgrounds?.length ?? 0) +
        " backgrounds · " +
        (character.images?.length ?? 0) +
        " images · " +
        (character.frames?.length ?? 0) +
        " frames"
      content.appendChild(counts)

      renderMark(content, character)
      card.appendChild(content)

      card.addEventListener("click", () => applyCharacter(character))
      list.appendChild(card)
    }
  }

  const updateSelection = () => {
    const current = getCurrent()
    if (!current) return

    current.textContent = getSelectedCharacter()?.name ?? "none"
  }

  const bindSettings = () => {
    const root = getRoot()
    if (!(root instanceof HTMLElement) || root.dataset.privateCharacterBound === "true") return
    root.dataset.privateCharacterBound = "true"

    document.addEventListener(CHARACTER_CHANGE_EVENT_NAME, () => {
      updateSelection()
      renderCharacters()
    })
  }

  const init = () => {
    bindSettings()
    updateSelection()
    renderCharacters()
  }

  document.addEventListener("nav", init)
  init()
})()
`

PrivateCharacterSettings.css = styles

export default (() => PrivateCharacterSettings) satisfies QuartzComponentConstructor
